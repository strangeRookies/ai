# Naver Blog Markdown Pipeline

정리본 Markdown을 블로그 글처럼 다듬고, `@jjlabsio/md-to-naver-blog`를 이용해 네이버 블로그 에디터에 붙여넣기 쉬운 HTML preview를 생성하는 작은 파이프라인입니다.

## 폴더 구조

```text
naver_blog_pipeline/
  blog_pipeline/              # Markdown 재구성 로직
  input/                      # 원본 정리본 Markdown 예시
  output/                     # 블로그용 Markdown, preview HTML 예시
  scripts/
    convert_to_blog.py        # 원본 Markdown -> 블로그용 Markdown
    publish_to_naver.py       # 블로그용 Markdown -> preview HTML/JSON
    md-to-naver-blog-preview.mjs
  tests/                      # 변환 로직 테스트
  .env.example                # 네이버 발행 자동화 확장 시 필요한 환경변수 이름
  package.json                # md-to-naver-blog 의존성
```

## 오픈소스 역할

- `im-not-ai`: AI 티가 나는 한글 표현을 줄이는 윤문 규칙/스킬입니다. 현재 파이프라인의 **전처리 단계에 통합**되어, 블로그 글로 변환하기 전에 기계적인 나열("첫째/둘째"), 번역투("~에 있어서"), AI 관용구("결론적으로", "시사하는 바가 크다")를 자연스럽게 순화합니다. LLM 연동 시에는 더 자연스러운 윤문을 제공하며, LLM 없이도 정규식 기반의 치환 규칙(Fallback)이 기본 작동합니다.
- `@jjlabsio/md-to-naver-blog`: Markdown을 네이버 블로그에 붙여넣기 좋은 HTML로 바꾸는 JS 라이브러리입니다. 로그인이나 실제 발행 API는 제공하지 않습니다.

## 실행 방법

### 1. 간편 실행 (추천)
마크다운(`.md`) 파일을 `input` 폴더에 넣고, 루트 폴더의 **`run.bat`** 파일을 더블 클릭하여 실행합니다.
- `input` 폴더에 있는 모든 `.md` 파일을 자동으로 변환합니다.
- 변환 완료 후 생성된 HTML 프리뷰 파일(`output/*.preview.html`)이 웹 브라우저에서 자동으로 열립니다.

### 2. 수동 실행
이 폴더에서 실행합니다.

```bash
cd naver_blog_pipeline
npm install

# 기본 실행 (im-not-ai 윤문 규칙 적용)
python scripts/convert_to_blog.py \
  --input input/my_note.md \
  --output output/blog_post.md \
  --humanize

# 윤문 없이 변환할 경우
# python scripts/convert_to_blog.py --input input/my_note.md --output output/blog_post.md --no-humanize

python scripts/publish_to_naver.py \
  --input output/blog_post.md \
  --draft
```

생성 파일:

- `output/blog_post.md`: 사람이 수정할 수 있는 중간 Markdown 산출물
- `output/blog_post.preview.html`: 업로드 전 확인용 HTML
- `output/blog_post.naver.json`: `md-to-naver-blog` 변환 제목, frontmatter, 파서 경고

## Preview 확인

`output/blog_post.preview.html`을 브라우저로 열어 결과를 확인합니다. 이미지 경로는 Markdown의 상대 경로를 유지하되 Windows 백슬래시는 `/`로 정규화합니다. 실제 네이버 에디터의 이미지 처리 방식은 계정/브라우저 환경에 따라 다를 수 있으므로 최종 업로드 전 직접 확인해야 합니다.

## 실제 발행 주의사항

현재 `@jjlabsio/md-to-naver-blog`는 HTML 변환 라이브러리이며 네이버 로그인/발행 기능을 제공하지 않습니다. 그래서 `scripts/publish_to_naver.py --publish`는 의도적으로 실패합니다.

향후 발행 자동화를 붙일 경우에도 `NAVER_BLOG_ID`, `NAVER_USERNAME`, `NAVER_PASSWORD`, `NAVER_SESSION_COOKIE` 같은 값은 `.env` 또는 환경변수로만 전달하고 코드에 하드코딩하지 마세요.
