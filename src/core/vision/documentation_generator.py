"""Documentation Generator Module for Vision System.

This module provides automatic documentation generation capabilities including:
- API documentation generation
- Code documentation extraction
- Markdown/HTML documentation output
- Interactive documentation (like Swagger UI)
- Change log generation
- Usage examples generation

Phase 16: Advanced Integration & Extensibility
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, get_type_hints

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class DocFormat(str, Enum):
    """Documentation output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"
    ASCIIDOC = "asciidoc"
    JSON = "json"
    YAML = "yaml"


class DocSection(str, Enum):
    """Documentation sections."""

    OVERVIEW = "overview"
    INSTALLATION = "installation"
    QUICKSTART = "quickstart"
    API_REFERENCE = "api_reference"
    CONFIGURATION = "configuration"
    EXAMPLES = "examples"
    CHANGELOG = "changelog"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    CONTRIBUTING = "contributing"
    LICENSE = "license"


class ChangeType(str, Enum):
    """Types of changes in changelog."""

    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    FIXED = "fixed"
    SECURITY = "security"


class ParameterType(str, Enum):
    """Parameter types for documentation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    ANY = "any"


# ========================
# Data Classes
# ========================


@dataclass
class ParameterDoc:
    """Documentation for a parameter."""

    name: str
    param_type: ParameterType
    description: str = ""
    required: bool = False
    default: Any = None
    example: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReturnDoc:
    """Documentation for return value."""

    return_type: str
    description: str = ""
    example: Any = None


@dataclass
class ExceptionDoc:
    """Documentation for exception."""

    exception_type: str
    description: str = ""
    conditions: str = ""


@dataclass
class MethodDoc:
    """Documentation for a method."""

    name: str
    summary: str = ""
    description: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    exceptions: List[ExceptionDoc] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str = ""
    since_version: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ClassDoc:
    """Documentation for a class."""

    name: str
    summary: str = ""
    description: str = ""
    methods: List[MethodDoc] = field(default_factory=list)
    properties: List[ParameterDoc] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    parent_classes: List[str] = field(default_factory=list)
    since_version: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ModuleDoc:
    """Documentation for a module."""

    name: str
    summary: str = ""
    description: str = ""
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[MethodDoc] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    version: str = ""


@dataclass
class ChangeLogEntry:
    """Entry in changelog."""

    version: str
    date: str
    change_type: ChangeType
    description: str
    breaking: bool = False
    issue_refs: List[str] = field(default_factory=list)
    author: str = ""


@dataclass
class ChangeLog:
    """Complete changelog."""

    entries: List[ChangeLogEntry] = field(default_factory=list)
    unreleased: List[ChangeLogEntry] = field(default_factory=list)


@dataclass
class DocConfig:
    """Documentation generation configuration."""

    title: str
    version: str
    description: str = ""
    author: str = ""
    format: DocFormat = DocFormat.MARKDOWN
    output_dir: str = "docs"
    include_private: bool = False
    include_source: bool = False
    include_examples: bool = True
    include_changelog: bool = True
    base_url: str = ""
    logo_url: str = ""
    theme: str = "default"
    custom_css: str = ""
    sections: List[DocSection] = field(
        default_factory=lambda: [
            DocSection.OVERVIEW,
            DocSection.INSTALLATION,
            DocSection.QUICKSTART,
            DocSection.API_REFERENCE,
            DocSection.EXAMPLES,
            DocSection.CHANGELOG,
        ]
    )


@dataclass
class GeneratedDoc:
    """Generated documentation output."""

    path: str
    content: str
    format: DocFormat
    section: Optional[DocSection] = None
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            import hashlib

            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class DocGenerationResult:
    """Result of documentation generation."""

    success: bool
    docs: List[GeneratedDoc] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


# ========================
# Documentation Extractors
# ========================


class DocstringParser:
    """Parser for Python docstrings."""

    def parse(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Parse a docstring into structured data."""
        if not docstring:
            return {"summary": "", "description": "", "params": [], "returns": None, "raises": []}

        lines = docstring.strip().split("\n")
        result = {
            "summary": "",
            "description": "",
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }

        # Get summary (first line)
        if lines:
            result["summary"] = lines[0].strip()

        # Parse sections
        current_section = "description"
        current_content: List[str] = []
        current_param: Optional[Dict[str, Any]] = None

        for i, line in enumerate(lines[1:], 1):
            stripped = line.strip()

            # Check for section headers
            if stripped.lower().startswith("args:") or stripped.lower().startswith("parameters:"):
                result["description"] = "\n".join(current_content).strip()
                current_section = "params"
                current_content = []
                continue

            if stripped.lower().startswith("returns:"):
                self._save_param(current_param, result)
                current_param = None
                current_section = "returns"
                current_content = []
                continue

            if stripped.lower().startswith("raises:") or stripped.lower().startswith("exceptions:"):
                self._save_param(current_param, result)
                current_param = None
                if current_section == "returns" and current_content:
                    result["returns"] = {
                        "type": "",
                        "description": "\n".join(current_content).strip(),
                    }
                current_section = "raises"
                current_content = []
                continue

            if stripped.lower().startswith("examples:") or stripped.lower().startswith("example:"):
                self._save_param(current_param, result)
                current_param = None
                current_section = "examples"
                current_content = []
                continue

            # Process content based on section
            if current_section == "params":
                # Check for new param (name: description format)
                param_match = re.match(r"(\w+)\s*(?:\([^)]+\))?\s*:\s*(.*)", stripped)
                if param_match and not stripped.startswith(" "):
                    self._save_param(current_param, result)
                    current_param = {
                        "name": param_match.group(1),
                        "description": param_match.group(2),
                    }
                elif current_param and stripped:
                    current_param["description"] += " " + stripped
            elif current_section == "raises":
                exc_match = re.match(r"(\w+)\s*:\s*(.*)", stripped)
                if exc_match:
                    result["raises"].append(
                        {"type": exc_match.group(1), "description": exc_match.group(2)}
                    )
            else:
                if stripped or current_content:
                    current_content.append(stripped)

        # Save any remaining content
        self._save_param(current_param, result)

        if current_section == "description":
            result["description"] = "\n".join(current_content).strip()
        elif current_section == "returns" and current_content:
            result["returns"] = {"type": "", "description": "\n".join(current_content).strip()}
        elif current_section == "examples":
            result["examples"] = current_content

        return result

    def _save_param(self, param: Optional[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """Save parameter to result."""
        if param:
            result["params"].append(param)


class CodeExtractor:
    """Extracts documentation from Python code."""

    def __init__(self):
        """Initialize code extractor."""
        self._parser = DocstringParser()

    def extract_module(self, module: Any) -> ModuleDoc:
        """Extract documentation from a module."""
        doc = ModuleDoc(
            name=module.__name__,
            summary=self._get_summary(module.__doc__),
            description=self._get_description(module.__doc__),
        )

        # Extract classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.startswith("_") and obj.__module__ == module.__name__:
                class_doc = self.extract_class(obj)
                doc.classes.append(class_doc)

        # Extract functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_") and obj.__module__ == module.__name__:
                method_doc = self.extract_method(obj)
                doc.functions.append(method_doc)

        return doc

    def extract_class(self, cls: Type) -> ClassDoc:
        """Extract documentation from a class."""
        parsed = self._parser.parse(cls.__doc__)

        doc = ClassDoc(
            name=cls.__name__,
            summary=parsed["summary"],
            description=parsed["description"],
            parent_classes=[base.__name__ for base in cls.__bases__ if base != object],
        )

        # Extract methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_") or name in ("__init__", "__call__"):
                method_doc = self.extract_method(method)
                doc.methods.append(method_doc)

        # Extract properties
        for name, prop in inspect.getmembers(cls, predicate=lambda x: isinstance(x, property)):
            if not name.startswith("_"):
                prop_doc = self._extract_property(name, prop)
                doc.properties.append(prop_doc)

        return doc

    def extract_method(self, method: Callable) -> MethodDoc:
        """Extract documentation from a method."""
        parsed = self._parser.parse(method.__doc__)

        doc = MethodDoc(
            name=method.__name__, summary=parsed["summary"], description=parsed["description"]
        )

        # Extract parameters
        sig = inspect.signature(method)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_info = next((p for p in parsed["params"] if p["name"] == param_name), {})

            param_doc = ParameterDoc(
                name=param_name,
                param_type=self._get_param_type(param),
                description=param_info.get("description", ""),
                required=param.default == inspect.Parameter.empty,
                default=None if param.default == inspect.Parameter.empty else param.default,
            )
            doc.parameters.append(param_doc)

        # Extract return type
        if parsed["returns"]:
            doc.returns = ReturnDoc(
                return_type=parsed["returns"].get("type", "Any"),
                description=parsed["returns"].get("description", ""),
            )

        # Extract exceptions
        for exc in parsed["raises"]:
            doc.exceptions.append(
                ExceptionDoc(exception_type=exc["type"], description=exc["description"])
            )

        return doc

    def _extract_property(self, name: str, prop: property) -> ParameterDoc:
        """Extract documentation from a property."""
        doc_str = prop.fget.__doc__ if prop.fget else ""
        return ParameterDoc(
            name=name, param_type=ParameterType.ANY, description=self._get_summary(doc_str)
        )

    def _get_summary(self, docstring: Optional[str]) -> str:
        """Get summary from docstring."""
        if not docstring:
            return ""
        lines = docstring.strip().split("\n")
        return lines[0].strip() if lines else ""

    def _get_description(self, docstring: Optional[str]) -> str:
        """Get description from docstring."""
        parsed = self._parser.parse(docstring)
        return parsed["description"]

    def _get_param_type(self, param: inspect.Parameter) -> ParameterType:
        """Get parameter type."""
        if param.annotation == inspect.Parameter.empty:
            return ParameterType.ANY

        type_str = str(param.annotation)

        if "str" in type_str:
            return ParameterType.STRING
        if "int" in type_str:
            return ParameterType.INTEGER
        if "float" in type_str:
            return ParameterType.FLOAT
        if "bool" in type_str:
            return ParameterType.BOOLEAN
        if "List" in type_str or "list" in type_str:
            return ParameterType.ARRAY
        if "Dict" in type_str or "dict" in type_str:
            return ParameterType.OBJECT

        return ParameterType.ANY


# ========================
# Document Generators
# ========================


class DocumentGenerator(ABC):
    """Abstract base class for document generators."""

    @abstractmethod
    def generate(self, doc: Union[ModuleDoc, ClassDoc, MethodDoc], config: DocConfig) -> str:
        """Generate documentation output."""
        pass


class MarkdownGenerator(DocumentGenerator):
    """Markdown documentation generator."""

    def generate(self, doc: Union[ModuleDoc, ClassDoc, MethodDoc], config: DocConfig) -> str:
        """Generate Markdown documentation."""
        if isinstance(doc, ModuleDoc):
            return self._generate_module(doc, config)
        if isinstance(doc, ClassDoc):
            return self._generate_class(doc, config)
        if isinstance(doc, MethodDoc):
            return self._generate_method(doc, config)
        return ""

    def _generate_module(self, doc: ModuleDoc, config: DocConfig) -> str:
        """Generate module documentation."""
        lines = [
            f"# {doc.name}",
            "",
            doc.summary,
            "",
        ]

        if doc.description:
            lines.extend([doc.description, ""])

        if doc.version:
            lines.extend([f"**Version:** {doc.version}", ""])

        # Table of contents
        if doc.classes or doc.functions:
            lines.extend(["## Table of Contents", ""])

            if doc.classes:
                lines.append("### Classes")
                for cls in doc.classes:
                    lines.append(f"- [{cls.name}](#{cls.name.lower()})")
                lines.append("")

            if doc.functions:
                lines.append("### Functions")
                for func in doc.functions:
                    lines.append(f"- [{func.name}](#{func.name.lower()})")
                lines.append("")

        # Classes
        for cls in doc.classes:
            lines.extend(self._generate_class(cls, config).split("\n"))
            lines.append("")

        # Functions
        if doc.functions:
            lines.extend(["## Functions", ""])
            for func in doc.functions:
                lines.extend(self._generate_method(func, config, heading_level=3).split("\n"))
                lines.append("")

        return "\n".join(lines)

    def _generate_class(self, doc: ClassDoc, config: DocConfig) -> str:
        """Generate class documentation."""
        lines = [
            f"## {doc.name}",
            "",
            doc.summary,
            "",
        ]

        if doc.description:
            lines.extend([doc.description, ""])

        if doc.parent_classes:
            lines.extend([f"**Inherits from:** {', '.join(doc.parent_classes)}", ""])

        if doc.since_version:
            lines.extend([f"**Since:** {doc.since_version}", ""])

        # Properties
        if doc.properties:
            lines.extend(["### Properties", ""])
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for prop in doc.properties:
                lines.append(f"| `{prop.name}` | {prop.param_type.value} | {prop.description} |")
            lines.append("")

        # Methods
        if doc.methods:
            lines.extend(["### Methods", ""])
            for method in doc.methods:
                lines.extend(self._generate_method(method, config, heading_level=4).split("\n"))
                lines.append("")

        # Examples
        if config.include_examples and doc.examples:
            lines.extend(["### Examples", ""])
            for example in doc.examples:
                lines.extend(["```python", example, "```", ""])

        return "\n".join(lines)

    def _generate_method(self, doc: MethodDoc, config: DocConfig, heading_level: int = 3) -> str:
        """Generate method documentation."""
        heading = "#" * heading_level
        lines = [
            f"{heading} {doc.name}",
            "",
            doc.summary,
            "",
        ]

        if doc.deprecated:
            lines.extend(
                [f"> **Deprecated:** {doc.deprecation_message or 'This method is deprecated.'}", ""]
            )

        if doc.description:
            lines.extend([doc.description, ""])

        # Signature
        params_str = ", ".join(
            f"{p.name}: {p.param_type.value}"
            + (f" = {p.default!r}" if p.default is not None else "")
            for p in doc.parameters
        )
        return_str = f" -> {doc.returns.return_type}" if doc.returns else ""
        lines.extend(["```python", f"def {doc.name}({params_str}){return_str}", "```", ""])

        # Parameters
        if doc.parameters:
            lines.extend(["**Parameters:**", ""])
            for param in doc.parameters:
                req = " *(required)*" if param.required else ""
                default = f" (default: `{param.default!r}`)" if param.default is not None else ""
                lines.append(
                    f"- `{param.name}` ({param.param_type.value}){req}: {param.description}{default}"
                )
            lines.append("")

        # Returns
        if doc.returns:
            lines.extend(
                [
                    "**Returns:**",
                    "",
                    f"- `{doc.returns.return_type}`: {doc.returns.description}",
                    "",
                ]
            )

        # Exceptions
        if doc.exceptions:
            lines.extend(["**Raises:**", ""])
            for exc in doc.exceptions:
                lines.append(f"- `{exc.exception_type}`: {exc.description}")
            lines.append("")

        # Examples
        if config.include_examples and doc.examples:
            lines.extend(["**Example:**", ""])
            for example in doc.examples:
                lines.extend(["```python", example, "```", ""])

        return "\n".join(lines)

    def generate_changelog(self, changelog: ChangeLog, config: DocConfig) -> str:
        """Generate changelog documentation."""
        lines = [
            "# Changelog",
            "",
            "All notable changes to this project will be documented in this file.",
            "",
        ]

        # Unreleased
        if changelog.unreleased:
            lines.extend(["## [Unreleased]", ""])
            lines.extend(self._format_entries(changelog.unreleased))

        # Group by version
        versions: Dict[str, List[ChangeLogEntry]] = {}
        for entry in changelog.entries:
            if entry.version not in versions:
                versions[entry.version] = []
            versions[entry.version].append(entry)

        for version, entries in versions.items():
            date = entries[0].date if entries else ""
            lines.extend([f"## [{version}] - {date}", ""])
            lines.extend(self._format_entries(entries))

        return "\n".join(lines)

    def _format_entries(self, entries: List[ChangeLogEntry]) -> List[str]:
        """Format changelog entries."""
        lines = []

        # Group by change type
        by_type: Dict[ChangeType, List[ChangeLogEntry]] = {}
        for entry in entries:
            if entry.change_type not in by_type:
                by_type[entry.change_type] = []
            by_type[entry.change_type].append(entry)

        for change_type, type_entries in by_type.items():
            lines.extend([f"### {change_type.value.capitalize()}", ""])
            for entry in type_entries:
                breaking = " **BREAKING**" if entry.breaking else ""
                refs = f" ({', '.join(entry.issue_refs)})" if entry.issue_refs else ""
                lines.append(f"- {entry.description}{breaking}{refs}")
            lines.append("")

        return lines


class HTMLGenerator(DocumentGenerator):
    """HTML documentation generator."""

    def generate(self, doc: Union[ModuleDoc, ClassDoc, MethodDoc], config: DocConfig) -> str:
        """Generate HTML documentation."""
        # Generate markdown first, then convert to HTML
        md_gen = MarkdownGenerator()
        markdown = md_gen.generate(doc, config)
        return self._markdown_to_html(markdown, config)

    def _markdown_to_html(self, markdown: str, config: DocConfig) -> str:
        """Convert Markdown to HTML (simplified)."""
        html = markdown

        # Convert headers
        for i in range(6, 0, -1):
            pattern = r"^" + "#" * i + r"\s+(.+)$"
            html = re.sub(pattern, f"<h{i}>\\1</h{i}>", html, flags=re.MULTILINE)

        # Convert code blocks
        html = re.sub(
            r"```(\w+)?\n(.*?)\n```",
            r'<pre><code class="\1">\2</code></pre>',
            html,
            flags=re.DOTALL,
        )

        # Convert inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Convert bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Convert italic
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Convert links
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

        # Convert lists
        lines = html.split("\n")
        in_list = False
        result_lines = []

        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    result_lines.append("<ul>")
                    in_list = True
                result_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result_lines.append("</ul>")
                    in_list = False
                result_lines.append(line)

        if in_list:
            result_lines.append("</ul>")

        html = "\n".join(result_lines)

        # Convert paragraphs
        html = re.sub(r"\n\n+", "</p>\n<p>", html)

        # Wrap in template
        template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 2rem; }}
        code {{ background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 1rem; border-radius: 5px; overflow-x: auto; }}
        pre code {{ background: none; padding: 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
        th {{ background: #f4f4f4; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 1rem; color: #666; }}
        {config.custom_css}
    </style>
</head>
<body>
<p>{html}</p>
</body>
</html>"""

        return template


# ========================
# Documentation Manager
# ========================


class DocumentationGenerator:
    """Main documentation generator orchestrator."""

    def __init__(self):
        """Initialize documentation generator."""
        self._lock = threading.Lock()
        self._generators: Dict[DocFormat, DocumentGenerator] = {
            DocFormat.MARKDOWN: MarkdownGenerator(),
            DocFormat.HTML: HTMLGenerator(),
        }
        self._extractor = CodeExtractor()
        self._changelog = ChangeLog()
        self._generation_history: List[DocGenerationResult] = []

    def register_generator(self, format: DocFormat, generator: DocumentGenerator) -> None:
        """Register a document generator."""
        with self._lock:
            self._generators[format] = generator

    def extract_documentation(self, module: Any) -> ModuleDoc:
        """Extract documentation from a module."""
        return self._extractor.extract_module(module)

    def generate(
        self, doc: Union[ModuleDoc, ClassDoc, MethodDoc], config: DocConfig
    ) -> DocGenerationResult:
        """Generate documentation."""
        generator = self._generators.get(config.format)
        if not generator:
            return DocGenerationResult(
                success=False, errors=[f"No generator for format: {config.format}"]
            )

        try:
            docs: List[GeneratedDoc] = []
            errors: List[str] = []
            warnings: List[str] = []

            # Generate main documentation
            content = generator.generate(doc, config)

            ext = {DocFormat.MARKDOWN: ".md", DocFormat.HTML: ".html", DocFormat.RST: ".rst"}.get(
                config.format, ".txt"
            )

            docs.append(
                GeneratedDoc(
                    path=f"{config.output_dir}/{doc.name}{ext}",
                    content=content,
                    format=config.format,
                    section=DocSection.API_REFERENCE,
                )
            )

            # Generate changelog if enabled
            if config.include_changelog and self._changelog.entries:
                if isinstance(generator, MarkdownGenerator):
                    changelog_content = generator.generate_changelog(self._changelog, config)
                    docs.append(
                        GeneratedDoc(
                            path=f"{config.output_dir}/CHANGELOG{ext}",
                            content=changelog_content,
                            format=config.format,
                            section=DocSection.CHANGELOG,
                        )
                    )

            stats = {
                "total_files": len(docs),
                "total_size": sum(len(d.content) for d in docs),
                "format": config.format.value,
            }

            result = DocGenerationResult(
                success=True, docs=docs, errors=errors, warnings=warnings, stats=stats
            )

        except Exception as e:
            result = DocGenerationResult(success=False, errors=[str(e)])

        with self._lock:
            self._generation_history.append(result)

        return result

    def add_changelog_entry(
        self,
        version: str,
        change_type: ChangeType,
        description: str,
        date: Optional[str] = None,
        breaking: bool = False,
        issue_refs: Optional[List[str]] = None,
    ) -> None:
        """Add a changelog entry."""
        entry = ChangeLogEntry(
            version=version,
            date=date or datetime.now().strftime("%Y-%m-%d"),
            change_type=change_type,
            description=description,
            breaking=breaking,
            issue_refs=issue_refs or [],
        )

        with self._lock:
            if version == "unreleased":
                self._changelog.unreleased.append(entry)
            else:
                self._changelog.entries.append(entry)

    def get_changelog(self) -> ChangeLog:
        """Get the changelog."""
        with self._lock:
            return self._changelog

    def get_generation_history(self) -> List[DocGenerationResult]:
        """Get generation history."""
        with self._lock:
            return list(self._generation_history)

    def generate_api_reference(self, modules: List[Any], config: DocConfig) -> DocGenerationResult:
        """Generate API reference for multiple modules."""
        all_docs: List[GeneratedDoc] = []
        errors: List[str] = []

        for module in modules:
            try:
                module_doc = self.extract_documentation(module)
                result = self.generate(module_doc, config)

                if result.success:
                    all_docs.extend(result.docs)
                else:
                    errors.extend(result.errors)

            except Exception as e:
                errors.append(f"Error processing {module.__name__}: {str(e)}")

        # Generate index
        index_content = self._generate_index(modules, config)
        ext = {DocFormat.MARKDOWN: ".md", DocFormat.HTML: ".html"}.get(config.format, ".txt")

        all_docs.insert(
            0,
            GeneratedDoc(
                path=f"{config.output_dir}/index{ext}",
                content=index_content,
                format=config.format,
                section=DocSection.OVERVIEW,
            ),
        )

        return DocGenerationResult(
            success=len(errors) == 0,
            docs=all_docs,
            errors=errors,
            stats={"modules_processed": len(modules), "total_files": len(all_docs)},
        )

    def _generate_index(self, modules: List[Any], config: DocConfig) -> str:
        """Generate index page."""
        lines = [
            f"# {config.title}",
            "",
            config.description,
            "",
            f"**Version:** {config.version}",
            "",
            "## Modules",
            "",
        ]

        for module in modules:
            name = module.__name__
            summary = self._extractor._get_summary(module.__doc__)
            lines.append(f"- [{name}]({name}.md): {summary}")

        return "\n".join(lines)


# ========================
# Vision Provider
# ========================


class DocumentedVisionProvider(VisionProvider):
    """Vision provider with documentation capabilities."""

    def __init__(
        self, base_provider: VisionProvider, doc_generator: Optional[DocumentationGenerator] = None
    ):
        """Initialize documented vision provider."""
        self._base_provider = base_provider
        self._doc_generator = doc_generator or DocumentationGenerator()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"documented_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with documentation support."""
        return await self._base_provider.analyze_image(
            image_data, include_description=include_description, **kwargs
        )

    def generate_documentation(self, module: Any, config: DocConfig) -> DocGenerationResult:
        """Generate documentation for a module."""
        module_doc = self._doc_generator.extract_documentation(module)
        return self._doc_generator.generate(module_doc, config)

    def add_changelog_entry(
        self, version: str, change_type: ChangeType, description: str, **kwargs: Any
    ) -> None:
        """Add changelog entry."""
        self._doc_generator.add_changelog_entry(version, change_type, description, **kwargs)


# ========================
# Factory Functions
# ========================


def create_documentation_generator() -> DocumentationGenerator:
    """Create a new documentation generator instance."""
    return DocumentationGenerator()


def create_markdown_generator() -> MarkdownGenerator:
    """Create a Markdown generator."""
    return MarkdownGenerator()


def create_html_generator() -> HTMLGenerator:
    """Create an HTML generator."""
    return HTMLGenerator()


def create_code_extractor() -> CodeExtractor:
    """Create a code extractor."""
    return CodeExtractor()


def create_docstring_parser() -> DocstringParser:
    """Create a docstring parser."""
    return DocstringParser()


def create_documented_provider(
    base_provider: VisionProvider, doc_generator: Optional[DocumentationGenerator] = None
) -> DocumentedVisionProvider:
    """Create a documented vision provider."""
    return DocumentedVisionProvider(base_provider, doc_generator)
