import tempfile
import unittest
from pathlib import Path

from naver_blog_pipeline.blog_pipeline.markdown_blog import (
    build_blog_draft,
    write_blog_draft,
    _protect_and_humanize,
)


class FakeLLMClient:
    def __init__(self, return_value: str = "", provider: str = "gemini"):
        self.provider = provider
        self.return_value = return_value
        self.received_text = ""

    def is_enabled(self) -> bool:
        return True

    def humanize(self, text: str) -> str:
        self.received_text = text
        return self.return_value


class BlogPipelineTest(unittest.TestCase):
    def test_build_blog_draft_preserves_code_blocks_and_commands(self):
        source = """# GPU PC RTSP 채널 정리

## 실행 명령

아래 명령으로 실행한다.

```bash
python scripts/run_registered_cameras.py --backend-base-url "http://127.0.0.1:18080"
```
"""

        draft = build_blog_draft(source)

        self.assertEqual(draft.title, "GPU PC RTSP 채널 정리")
        self.assertIn("## 도입부", draft.markdown)
        self.assertIn("## 핵심 정리", draft.markdown)
        self.assertIn(
            'python scripts/run_registered_cameras.py --backend-base-url "http://127.0.0.1:18080"',
            draft.markdown,
        )

    def test_build_blog_draft_keeps_image_markdown(self):
        source = """# 대시보드 개선 기록

## 화면 예시

![대시보드](./images/dashboard.png)
"""

        draft = build_blog_draft(source)

        self.assertIn("![대시보드](./images/dashboard.png)", draft.markdown)
        self.assertIn("- 화면 예시 내용을 중심으로 확인합니다.", draft.markdown)

    def test_write_blog_draft_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input" / "note.md"
            output_path = root / "output" / "blog.md"
            input_path.parent.mkdir()
            input_path.write_text("# 테스트 정리\n\n본문입니다.\n", encoding="utf-8")

            draft = write_blog_draft(input_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertIn(draft.title, output_path.read_text(encoding="utf-8"))

    def test_protect_and_humanize_placeholder_replacement_and_restoration(self):
        text = "여기에 `python script.py`가 있습니다. 자세한 건 [가이드](http://guide.com)와 ![이미지](./img.png)를 보세요."
        
        # We expect placeholders like __CODE_0__, __LINK_1__, __IMG_2__ (or order)
        # Note: in code, IMAGE is replaced first, then LINK, then CODE.
        # Length of placeholders: IMAGE is first -> len is 0 -> __IMG_0__
        # LINK is second -> len is 1 -> __LINK_1__
        # CODE is third -> len is 2 -> __CODE_2__
        expected_processed = "여기에 __CODE_2__가 있습니다. 자세한 건 __LINK_1__와 __IMG_0__를 보세요."
        
        fake_response = "여기 __CODE_2__가 있으며, 더 자세히 확인하려면 __LINK_1__와 __IMG_0__를 참고해 주세요."
        fake_client = FakeLLMClient(return_value=fake_response)
        
        result = _protect_and_humanize(text, fake_client)
        
        self.assertEqual(fake_client.received_text, expected_processed)
        self.assertIn("`python script.py`", result)
        self.assertIn("[가이드](http://guide.com)", result)
        self.assertIn("![이미지](./img.png)", result)
        self.assertEqual(result, "여기 `python script.py`가 있으며, 더 자세히 확인하려면 [가이드](http://guide.com)와 ![이미지](./img.png)를 참고해 주세요.")

    def test_protect_and_humanize_fallback_when_placeholders_are_lost(self):
        text = "코드 `hello`와 링크 [링크](http://url)가 있습니다."
        
        # LLM response with missing placeholder
        fake_response = "코드와 링크가 있습니다. (플레이스홀더를 유실함)"
        fake_client = FakeLLMClient(return_value=fake_response)
        
        # Should fallback to regular soften_paragraph
        result = _protect_and_humanize(text, fake_client)
        self.assertEqual(result, "코드 `hello`와 링크 [링크](http://url)가 있습니다.")


if __name__ == "__main__":
    unittest.main()
