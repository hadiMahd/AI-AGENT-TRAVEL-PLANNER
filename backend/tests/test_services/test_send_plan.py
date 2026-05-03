"""Send plan service tests — markdown→HTML conversion, Resend error handling."""

import pytest

from services.send_plan import _markdown_to_html, send_plan_email


class TestMarkdownToHTML:
    def test_headers_converted(self):
        result = _markdown_to_html("# Title\n\n## Section\n\n### Sub")
        assert "<h1>Title</h1>" in result
        assert "<h2>Section</h2>" in result
        assert "<h3>Sub</h3>" in result

    def test_bold_converted(self):
        result = _markdown_to_html("Hello **bold** world")
        assert "<strong>bold</strong>" in result

    def test_italic_converted(self):
        result = _markdown_to_html("Hello *italic* world")
        assert "<em>italic</em>" in result

    def test_bullet_list_converted(self):
        result = _markdown_to_html("* Item one\n* Item two")
        assert "<ul>" in result
        assert "<li>Item one</li>" in result
        assert "<li>Item two</li>" in result

    def test_plain_text_wrapped_in_paragraph(self):
        result = _markdown_to_html("Plain text paragraph")
        assert "<p>" in result

    def test_empty_string_returns_empty(self):
        result = _markdown_to_html("")
        assert result == ""


@pytest.mark.asyncio
async def test_send_plan_email_no_api_key(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    result = await send_plan_email("to@test.com", "plan content", "Bali")

    assert result is False


@pytest.mark.asyncio
async def test_send_plan_email_success(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "fake-key")

    import resend

    original = resend.Emails.send

    def fake_send(data):
        return {"id": "msg_test_123"}

    resend.Emails.send = fake_send

    try:
        result = await send_plan_email("to@test.com", "## Plan\n\nHello", "Bali")
        assert result is True
    finally:
        resend.Emails.send = original


@pytest.mark.asyncio
async def test_send_plan_email_failure_returns_false(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "fake-key")

    import resend

    original = resend.Emails.send

    def fake_send(data):
        raise Exception("API down")

    resend.Emails.send = fake_send

    try:
        result = await send_plan_email("to@test.com", "plan", "Bali")
        assert result is False
    finally:
        resend.Emails.send = original
