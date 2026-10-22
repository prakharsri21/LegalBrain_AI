import unittest

from app.services.response_cache import ResponseCache, build_cache_key


class ResponseCacheTests(unittest.TestCase):
    def test_cache_miss_then_hit(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=60)

        self.assertIsNone(cache.get("question-1"))

        cache.set("question-1", "Answer A")

        self.assertEqual(cache.get("question-1"), "Answer A")

        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_lru_eviction(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=60)

        cache.set("a", "Answer A")
        cache.set("b", "Answer B")

        # Access a, making b the least recently used entry.
        self.assertEqual(cache.get("a"), "Answer A")

        cache.set("c", "Answer C")

        self.assertEqual(cache.get("a"), "Answer A")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), "Answer C")

    def test_ttl_expiration(self):
        current_time = [100.0]

        cache = ResponseCache(
            max_entries=2,
            ttl_seconds=5,
            clock=lambda: current_time[0],
        )

        cache.set("a", "Answer A")
        self.assertEqual(cache.get("a"), "Answer A")

        current_time[0] = 106.0

        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.stats()["entries"], 0)

    def test_none_values_are_not_cached(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=60)

        cache.set("a", None)

        self.assertIsNone(cache.get("a"))

        stats = cache.stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 1)

    def test_same_request_creates_same_key(self):
        args = {
            "model": "Qwen/Qwen3-4B",
            "adapter": "base",
            "question": "What does Article 30 provide?",
            "max_tokens": 120,
            "temperature": 0.0,
            "prompt_version": "v1",
        }

        self.assertEqual(
            build_cache_key(**args),
            build_cache_key(**args),
        )

    def test_different_adapter_creates_different_key(self):
        args = {
            "model": "Qwen/Qwen3-4B",
            "adapter": "base",
            "question": "What does Article 30 provide?",
            "max_tokens": 120,
            "temperature": 0.0,
            "prompt_version": "v1",
        }

        base_key = build_cache_key(**args)

        args["adapter"] = "fine-tuned"
        fine_tuned_key = build_cache_key(**args)

        self.assertNotEqual(base_key, fine_tuned_key)

    def test_clear_removes_entries_and_resets_stats(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=60)

        cache.set("a", "Answer A")
        cache.get("a")
        cache.get("missing")

        cache.clear()

        stats = cache.stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)


if __name__ == "__main__":
    unittest.main()