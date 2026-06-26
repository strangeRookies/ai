import tempfile
import unittest
from pathlib import Path

from naver_blog_pipeline.blog_pipeline.markdown_blog import (
    build_blog_draft,
    write_blog_draft,
    _protect_and_humanize,
    _protect_and_generate_full,
    humanize_paragraph,
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

    def generate_full_post(self, text: str) -> str:
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

    def test_protect_and_generate_full_placeholder_replacement_and_restoration(self):
        text = "여기에 `python script.py`가 있습니다. 자세한 건 [가이드](http://guide.com)와 ![이미지](./img.png)를 보세요.\n\n```python\nprint('hello')\n```"
        
        # FENCE replaces first -> __FENCE_0__
        # IMG replaces second -> __IMG_1__
        # LINK replaces third -> __LINK_2__
        # CODE replaces fourth -> __CODE_3__
        expected_processed = "여기에 __CODE_3__가 있습니다. 자세한 건 __LINK_2__와 __IMG_1__를 보세요.\n\n__FENCE_0__"
        fake_response = "여기 __CODE_3__가 있으며, 더 자세히 확인하려면 __LINK_2__와 __IMG_1__를 참고해 주세요.\n\n그리고 코드:\n__FENCE_0__"
        fake_client = FakeLLMClient(return_value=fake_response)
        
        result = _protect_and_generate_full(text, fake_client)
        
        self.assertIn("`python script.py`", result)
        self.assertIn("[가이드](http://guide.com)", result)
        self.assertIn("![이미지](./img.png)", result)
        self.assertIn("```python", result)

    def test_protect_and_humanize_fallback_when_placeholders_are_lost(self):
        text = "코드 `hello`와 링크 [링크](http://url)가 있습니다."
        
        # LLM response with missing placeholder
        fake_response = "코드와 링크가 있습니다. (플레이스홀더를 유실함)"
        fake_client = FakeLLMClient(return_value=fake_response)
        
        # Should fallback to regular humanize_paragraph
        result = _protect_and_humanize(text, fake_client)
        self.assertEqual(result, "코드 `hello`와 링크 [링크](http://url)가 있습니다.")

    def test_humanize_paragraph_deterministic_rules(self):
        # 1. AI 관용구 제거 테스트
        text1 = "결론적으로, 이 기능은 아주 유용합니다."
        self.assertEqual(humanize_paragraph(text1), "이 기능은 아주 유용합니다.")

        text2 = "시사하는 바가 크다. 다음 단계로 가보자."
        self.assertEqual(humanize_paragraph(text2), "다음 단계로 가보자.")

        text3 = "첫째, 속도가 빠릅니다. 둘째, 안정적입니다."
        self.assertEqual(humanize_paragraph(text3), "속도가 빠릅니다. 안정적입니다.")

        # 2. 번역투 제거 테스트
        text4 = "성능 최적화에 있어서 중요한 부분입니다."
        self.assertEqual(humanize_paragraph(text4), "성능 최적화에서 중요한 부분입니다.")

        text5 = "캐싱을 통해 속도가 개선되어집니다."
        self.assertEqual(humanize_paragraph(text5), "캐싱(으)로 속도가 개선됩니다.")

        text6 = "외부 API에 의해 데이터가 처리되어지다."
        # 처리되어지다 -> 처리되다 -> 처리됩니다 (이다/된다/한다 -> 입니다/됩니다/합니다 적용은 안됨, 
        # 한다/이다/된다 가 아니라 되어지다 -> 되다 로 변경됨)
        # "외부 API(으)로 데이터가 처리되다."
        self.assertEqual(humanize_paragraph(text6), "외부 API(으)로 데이터가 처리되다.")

        # 3. 어미 자연화 테스트
        text7 = "기능이 동작한다. 데이터가 출력된다."
        self.assertEqual(humanize_paragraph(text7), "기능이 동작합니다. 데이터가 출력됩니다.")


if __name__ == "__main__":
    unittest.main()
