import requests
from openai import OpenAI
import re

class GithubAPI:
    def __init__(self, token):
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_pull_requests(self, owner, repo):
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise requests.exceptions.RequestException("Pull Request 목록 가져오기 실패")
        return response.json()

    def get_comments(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"코멘트 가져오기 실패: {url}")
            return []
        return response.json()

class OpenAIAPI:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",  # You can use an OpenAI key instead
            api_key=api_key,
        )

    def analyze_prs(self, messages):
        response = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": """
                다음은 Pull Request와 리뷰 목록입니다.
                PR 제목을 표시하고 그 밑에 설명을 요약하고 그 밑에 리뷰의 요약을 표시하세요.
                리뷰를 그대로 출력하는게 아닙니다. 요약을 해야합니다. 리뷰를 누가 했는지 명시적으로 표기해야 합니다.
                별도의 추가 코멘트는 필요하지 않습니다.
                """
            }] + messages,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4096,
            top_p=0.2,
        )
        return response.choices[0].message.content

class PRProcessor:
    def __init__(self, repo_owner, repo_name, github_token):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_api = GithubAPI(github_token)
        self.openai_api = OpenAIAPI(github_token)

    def process_prs(self):
        try:
            pulls = self.github_api.get_pull_requests(self.repo_owner, self.repo_name)
            messages = []
            
            for pr in pulls:
                messages.extend(self._process_single_pr(pr))
            
            return self.openai_api.analyze_prs(messages)
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생: {e}")
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")

    def _process_single_pr(self, pr):
        messages = []
        
        # PR 기본 정보 처리
        pr_info = self._extract_pr_info(pr)
        messages.append(self._create_pr_message(pr_info))
        
        # PR 코멘트 처리
        comments = self._get_comments(pr)
        for comment in comments:
            if not "[bot]" in comment["user"]["login"]:
                messages.append(self._create_comment_message(pr["number"], comment))
        
        return messages

    def _extract_pr_info(self, pr):
        summary = pr["body"] or ""
        summary = summary.replace("\r\n", "\n")

        return {
            "number": pr["number"],
            "title": pr["title"],
            "user": pr["user"]["login"],
            "summary": summary
        }

    def _get_comments(self, pr):
        comments = self.github_api.get_comments(pr["comments_url"])
        review_comments = self.github_api.get_comments(pr["review_comments_url"])
        all_comments = comments + review_comments
        return sorted(all_comments, key=lambda x: x["created_at"])

    def _create_pr_message(self, pr_info):
        return {
            "role": "user",
            "content": f"PR #{pr_info['number']}: {pr_info['title']} by {pr_info['user']}.\n{pr_info['summary']}"
        }

    def _create_comment_message(self, pr_number, comment):
        return {
            "role": "user",
            "content": f"#{pr_number} @{comment['user']['login']} commented: {comment['body']}"
        }

def main():
    # 설정값
    REPO_OWNER = "repo"
    REPO_NAME = "name"
    GITHUB_TOKEN = "yourtoken"

    # PR 처리 및 분석 실행
    processor = PRProcessor(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
    result = processor.process_prs()
    print(result)

if __name__ == "__main__":
    main()
