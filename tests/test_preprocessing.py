"""Tests for src/preprocessing.py."""

import pytest
from src.preprocessing import preprocess_text, preprocess_batch, post_from_twitter


class TestPreprocessText:
    def test_removes_urls(self):
        assert "http" not in preprocess_text("Check this out https://example.com now")

    def test_removes_mentions(self):
        assert "@user" not in preprocess_text("Hello @user how are you")

    def test_removes_hashtag_symbol_keeps_word(self):
        result = preprocess_text("I love #Python programming")
        assert "#" not in result
        assert "python" in result

    def test_removes_rt_prefix(self):
        assert not preprocess_text("RT: Hello world").startswith("rt")

    def test_collapses_repeated_characters(self):
        result = preprocess_text("I loooooove this!!!")
        assert "looo" not in result

    def test_lowercases_output(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_none_equivalent_returns_empty(self):
        assert preprocess_text("   ") == ""

    def test_normal_text_preserved(self):
        result = preprocess_text("I really enjoyed the movie.")
        assert "enjoyed" in result
        assert "movie" in result


class TestPreprocessBatch:
    def test_returns_list_same_length(self):
        texts = ["Hello world", "Test text", "Another one"]
        result = preprocess_batch(texts)
        assert len(result) == len(texts)

    def test_each_item_is_preprocessed(self):
        texts = ["@user check https://example.com", "#AI is great"]
        result = preprocess_batch(texts)
        assert all("@" not in r and "http" not in r for r in result)


class TestPostFromTwitter:
    def test_builds_post_correctly(self):
        tweet = {
            "id": "123",
            "text": "I love #MachineLearning!",
            "author_id": "u1",
            "created_at": "2026-06-01T10:00:00.000Z",
            "lang": "en",
        }
        post = post_from_twitter(tweet)
        assert post.post_id == "123"
        assert post.source  == "twitter"
        assert post.clean_text != ""
        assert "#" not in post.clean_text
