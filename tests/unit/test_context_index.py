"""
Tests for hierarchical file index.

Tests: SPEC-01.04 (Phase 4 - Hierarchical File Index)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.context_index import ContextIndex, FileIndex, IndexStats


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def context_index():
    """Create an in-memory context index."""
    return ContextIndex()


@pytest.fixture
def populated_index(temp_dir, context_index):
    """Create index with some test files."""
    # Create Python files
    py_file = temp_dir / "main.py"
    py_file.write_text(
        """
import os
import sys
from pathlib import Path

def main():
    print("Hello, world!")

class Application:
    def run(self):
        pass
"""
    )

    helper_file = temp_dir / "helper.py"
    helper_file.write_text(
        """
from typing import List

def helper_function(items: List[str]) -> int:
    return len(items)

class HelperClass:
    pass
"""
    )

    # Create JS file
    js_file = temp_dir / "app.js"
    js_file.write_text(
        """
import React from 'react';
import { useState } from 'react';

export function App() {
    return <div>Hello</div>;
}

export class Component {
    render() {}
}
"""
    )

    # Create markdown file
    md_file = temp_dir / "README.md"
    md_file.write_text("# Test Project\n\nThis is a test project.")

    # Index the directory
    context_index.index_directory(temp_dir)

    return context_index, temp_dir


class TestFileIndex:
    """Tests for FileIndex dataclass."""

    def test_file_index_creation(self):
        """FileIndex can be created with required fields."""
        idx = FileIndex(
            path="test.py",
            token_count=100,
            summary="Test file",
        )

        assert idx.path == "test.py"
        assert idx.token_count == 100
        assert idx.summary == "Test file"
        assert idx.imports == []
        assert idx.exports == []

    def test_file_index_with_all_fields(self):
        """FileIndex can be created with all fields."""
        idx = FileIndex(
            path="test.py",
            token_count=100,
            summary="Test file",
            imports=["os", "sys"],
            exports=["main", "App"],
            content_hash="abc123",
            mtime=1234567890.0,
        )

        assert idx.imports == ["os", "sys"]
        assert idx.exports == ["main", "App"]
        assert idx.content_hash == "abc123"
        assert idx.mtime == 1234567890.0


class TestContextIndexSchema:
    """Tests for ContextIndex schema creation."""

    def test_creates_tables(self, context_index):
        """Creates required tables."""
        # Use internal connection for in-memory database
        conn = context_index._get_connection()

        # Check file_index table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='file_index'"
        )
        assert cursor.fetchone() is not None

        # Check file_fts virtual table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='file_fts'"
        )
        assert cursor.fetchone() is not None

        context_index._close_connection(conn)


class TestIndexFile:
    """Tests for indexing individual files."""

    def test_index_python_file(self, context_index, temp_dir):
        """Index a Python file."""
        py_file = temp_dir / "test.py"
        py_file.write_text(
            """
import os

def hello():
    print("Hello")

class World:
    pass
"""
        )

        result = context_index.index_file(py_file, base_path=temp_dir)

        assert result is not None
        assert result.path == "test.py"
        assert result.token_count > 0
        assert "os" in result.imports
        assert "hello" in result.exports
        assert "World" in result.exports

    def test_index_javascript_file(self, context_index, temp_dir):
        """Index a JavaScript file."""
        js_file = temp_dir / "app.js"
        js_file.write_text(
            """
import React from 'react';

export function App() {
    return null;
}
"""
        )

        result = context_index.index_file(js_file, base_path=temp_dir)

        assert result is not None
        assert result.path == "app.js"
        assert "react" in result.imports
        assert "App" in result.exports

    def test_skip_non_indexable_extensions(self, context_index, temp_dir):
        """Skip files with non-indexable extensions."""
        binary_file = temp_dir / "image.png"
        binary_file.write_bytes(b"\x89PNG")

        result = context_index.index_file(binary_file, base_path=temp_dir)

        assert result is None

    def test_skip_nonexistent_file(self, context_index, temp_dir):
        """Skip nonexistent files."""
        result = context_index.index_file(temp_dir / "nonexistent.py")

        assert result is None

    def test_caches_unchanged_files(self, context_index, temp_dir):
        """Does not re-index unchanged files."""
        py_file = temp_dir / "test.py"
        py_file.write_text("def foo(): pass")

        # Index twice
        result1 = context_index.index_file(py_file, base_path=temp_dir)
        result2 = context_index.index_file(py_file, base_path=temp_dir)

        assert result1 is not None
        assert result2 is not None
        assert result1.content_hash == result2.content_hash


class TestIndexDirectory:
    """Tests for indexing directories."""

    def test_index_directory(self, context_index, temp_dir):
        """Index all files in directory."""
        # Create files
        (temp_dir / "a.py").write_text("def a(): pass")
        (temp_dir / "b.py").write_text("def b(): pass")
        (temp_dir / "c.js").write_text("function c() {}")

        count = context_index.index_directory(temp_dir)

        assert count == 3

    def test_skips_excluded_directories(self, context_index, temp_dir):
        """Skips node_modules and other excluded directories."""
        # Create file in node_modules
        node_modules = temp_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / "pkg.js").write_text("function pkg() {}")

        # Create file in src
        src = temp_dir / "src"
        src.mkdir()
        (src / "app.js").write_text("function app() {}")

        count = context_index.index_directory(temp_dir)

        assert count == 1  # Only src/app.js

    def test_respects_max_files(self, context_index, temp_dir):
        """Respects max_files limit."""
        for i in range(10):
            (temp_dir / f"file{i}.py").write_text(f"def f{i}(): pass")

        count = context_index.index_directory(temp_dir, max_files=5)

        assert count == 5

    def test_progress_callback(self, context_index, temp_dir):
        """Calls progress callback."""
        (temp_dir / "a.py").write_text("def a(): pass")
        (temp_dir / "b.py").write_text("def b(): pass")

        progress_calls = []

        def callback(indexed, total):
            progress_calls.append((indexed, total))

        context_index.index_directory(temp_dir, progress_callback=callback)

        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == 2


class TestSearch:
    """Tests for file search."""

    def test_search_by_content(self, populated_index):
        """Search files by content."""
        index, _ = populated_index

        results = index.search("hello")

        assert len(results) >= 1
        paths = [r.path for r in results]
        assert any("main.py" in p for p in paths)

    def test_search_by_path(self, populated_index):
        """Search files by path."""
        index, _ = populated_index

        results = index.search("helper")

        assert len(results) >= 1
        assert any("helper.py" in r.path for r in results)

    def test_search_by_exports(self, populated_index):
        """Search files by exported symbols."""
        index, _ = populated_index

        results = index.search("Application")

        assert len(results) >= 1
        paths = [r.path for r in results]
        assert any("main.py" in p for p in paths)

    def test_search_empty_query(self, populated_index):
        """Empty query returns no results."""
        index, _ = populated_index

        results = index.search("")

        assert results == []

    def test_search_no_matches(self, populated_index):
        """Non-matching query returns no results."""
        index, _ = populated_index

        results = index.search("xyznonexistent")

        assert results == []

    def test_search_limit(self, populated_index):
        """Search respects limit."""
        index, _ = populated_index

        results = index.search("def", limit=1)

        assert len(results) <= 1


class TestGetRelevantContext:
    """Tests for budget-aware context retrieval."""

    def test_get_relevant_context(self, populated_index):
        """Get relevant file contents within budget."""
        index, temp_dir = populated_index

        context = index.get_relevant_context("hello", budget_tokens=10000, base_path=temp_dir)

        assert len(context) >= 1
        assert any("main.py" in k for k in context.keys())

    def test_respects_token_budget(self, populated_index):
        """Respects token budget limit."""
        index, temp_dir = populated_index

        # Very small budget
        context = index.get_relevant_context("def", budget_tokens=10, base_path=temp_dir)

        # Should return few or no files due to small budget
        total_content = "".join(context.values())
        from src.tokenization import count_tokens

        # Files that fit should be within budget (or fallback to summary)
        assert count_tokens(total_content) <= 200  # Allow for summary fallback

    def test_empty_query_returns_empty(self, populated_index):
        """Empty query returns empty context."""
        index, temp_dir = populated_index

        context = index.get_relevant_context("", budget_tokens=10000, base_path=temp_dir)

        assert context == {}


class TestGetFile:
    """Tests for getting individual file entries."""

    def test_get_existing_file(self, populated_index):
        """Get existing file entry."""
        index, _ = populated_index

        result = index.get_file("main.py")

        assert result is not None
        assert result.path == "main.py"
        assert result.token_count > 0

    def test_get_nonexistent_file(self, populated_index):
        """Get nonexistent file returns None."""
        index, _ = populated_index

        result = index.get_file("nonexistent.py")

        assert result is None


class TestGetStats:
    """Tests for index statistics."""

    def test_get_stats(self, populated_index):
        """Get index statistics."""
        index, _ = populated_index

        stats = index.get_stats()

        assert isinstance(stats, IndexStats)
        assert stats.total_files == 4  # main.py, helper.py, app.js, README.md
        assert stats.total_tokens > 0
        assert stats.avg_tokens_per_file > 0
        assert len(stats.indexed_extensions) > 0

    def test_empty_index_stats(self, context_index):
        """Empty index returns zero stats."""
        stats = context_index.get_stats()

        assert stats.total_files == 0
        assert stats.total_tokens == 0
        assert stats.avg_tokens_per_file == 0


class TestClear:
    """Tests for clearing the index."""

    def test_clear_index(self, populated_index):
        """Clear all indexed files."""
        index, _ = populated_index

        count = index.clear()

        assert count == 4
        assert index.get_stats().total_files == 0


class TestRemoveFile:
    """Tests for removing individual files."""

    def test_remove_existing_file(self, populated_index):
        """Remove existing file from index."""
        index, _ = populated_index

        result = index.remove_file("main.py")

        assert result is True
        assert index.get_file("main.py") is None

    def test_remove_nonexistent_file(self, populated_index):
        """Remove nonexistent file returns False."""
        index, _ = populated_index

        result = index.remove_file("nonexistent.py")

        assert result is False


class TestRebuildFTSIndex:
    """Tests for rebuilding FTS index."""

    def test_rebuild_fts_index(self, populated_index):
        """Rebuild FTS index."""
        index, _ = populated_index

        # Verify initial search works
        assert len(index.search("hello")) >= 1

        # Rebuild the index (should work even on healthy index)
        count = index.rebuild_fts_index()
        assert count == 4

        # Search should still work after rebuild
        assert len(index.search("hello")) >= 1


class TestSummaryCreation:
    """Tests for file summary creation."""

    def test_python_summary_includes_structure(self, context_index, temp_dir):
        """Python file summary includes structure info."""
        py_file = temp_dir / "test.py"
        py_file.write_text(
            """
def func1(): pass
def func2(): pass
class Class1: pass
"""
        )

        result = context_index.index_file(py_file, base_path=temp_dir)

        assert result is not None
        assert "[Structure:" in result.summary
        assert "1 classes" in result.summary
        assert "2 functions" in result.summary

    def test_js_summary_includes_structure(self, context_index, temp_dir):
        """JavaScript file summary includes structure info."""
        js_file = temp_dir / "test.js"
        js_file.write_text(
            """
function func1() {}
function func2() {}
class Class1 {}
"""
        )

        result = context_index.index_file(js_file, base_path=temp_dir)

        assert result is not None
        assert "[Structure:" in result.summary


class TestImportExtraction:
    """Tests for import extraction."""

    def test_extract_python_imports(self, context_index, temp_dir):
        """Extract imports from Python files."""
        py_file = temp_dir / "test.py"
        py_file.write_text(
            """
import os
import sys
from pathlib import Path
from typing import List, Dict
"""
        )

        result = context_index.index_file(py_file, base_path=temp_dir)

        assert result is not None
        assert "os" in result.imports
        assert "sys" in result.imports
        assert "pathlib" in result.imports
        assert "typing" in result.imports

    def test_extract_js_imports(self, context_index, temp_dir):
        """Extract imports from JavaScript files."""
        js_file = temp_dir / "test.js"
        js_file.write_text(
            """
import React from 'react';
import { useState } from 'react';
import lodash from 'lodash';
"""
        )

        result = context_index.index_file(js_file, base_path=temp_dir)

        assert result is not None
        assert "react" in result.imports
        assert "lodash" in result.imports


class TestExportExtraction:
    """Tests for export extraction."""

    def test_extract_python_exports(self, context_index, temp_dir):
        """Extract exports from Python files."""
        py_file = temp_dir / "test.py"
        py_file.write_text(
            """
def public_func():
    pass

async def async_func():
    pass

class PublicClass:
    pass
"""
        )

        result = context_index.index_file(py_file, base_path=temp_dir)

        assert result is not None
        assert "public_func" in result.exports
        assert "async_func" in result.exports
        assert "PublicClass" in result.exports

    def test_extract_js_exports(self, context_index, temp_dir):
        """Extract exports from JavaScript files."""
        js_file = temp_dir / "test.js"
        js_file.write_text(
            """
export function exportedFunc() {}
export default function defaultFunc() {}
export class ExportedClass {}
export const exportedConst = 42;
"""
        )

        result = context_index.index_file(js_file, base_path=temp_dir)

        assert result is not None
        assert "exportedFunc" in result.exports
        assert "defaultFunc" in result.exports
        assert "ExportedClass" in result.exports
        assert "exportedConst" in result.exports
