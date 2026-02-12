from typing import List, Any

class Source:
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"Source(title={self.title}, url={self.url}, snippet={self.snippet})"

class SourceManager:
    def __init__(self):
        self.sources: List[Source] = []

    def add_source(self, title: str, url: str, snippet: str) -> None:
        source = Source(title, url, snippet)
        self.sources.append(source)

    def get_sources(self) -> List[Source]:
        return self.sources

    def clear_sources(self) -> None:
        self.sources.clear()