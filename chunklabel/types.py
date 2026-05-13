from dataclasses import dataclass


@dataclass
class RawChunk:
    category: str
    quote: str


@dataclass
class Chunk:
    category: str
    quote: str
    start: int
    end: int
