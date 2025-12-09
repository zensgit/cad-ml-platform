"""Custom prompt configuration for vision providers.

Provides:
- Prompt templates for different use cases
- User-configurable system prompts
- Prompt chaining and composition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of analysis prompts."""

    ENGINEERING_DRAWING = "engineering_drawing"
    ARCHITECTURAL = "architectural"
    CIRCUIT_DIAGRAM = "circuit_diagram"
    FLOWCHART = "flowchart"
    GENERAL = "general"
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""

    name: str
    system_prompt: str
    user_prompt_template: str
    output_format: str = "json"
    variables: List[str] = field(default_factory=list)
    description: str = ""

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """
        Render the template with variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.user_prompt_template
        for var in self.variables:
            placeholder = f"{{{var}}}"
            if placeholder in user_prompt:
                value = kwargs.get(var, "")
                user_prompt = user_prompt.replace(placeholder, str(value))

        return self.system_prompt, user_prompt


# Built-in prompt templates
ENGINEERING_DRAWING_PROMPT = PromptTemplate(
    name="engineering_drawing",
    description="Analyze engineering/mechanical drawings",
    system_prompt="""You are an expert engineering drawing analyst. Analyze the provided \
image and extract technical information.

Your analysis should identify:
1. Type of drawing (mechanical part, assembly, schematic, etc.)
2. Key dimensions and measurements
3. Materials and specifications if visible
4. Geometric features (holes, threads, chamfers, etc.)
5. Tolerances and surface finishes
6. Any notes, symbols, or annotations

Respond in JSON format with:
{
    "summary": "Brief description of the drawing",
    "details": ["List of specific observations"],
    "confidence": 0.0-1.0 confidence score
}""",
    user_prompt_template="Analyze this engineering drawing and provide technical details.",
    output_format="json",
)

ARCHITECTURAL_PROMPT = PromptTemplate(
    name="architectural",
    description="Analyze architectural drawings and floor plans",
    system_prompt="""You are an expert architectural drawing analyst. Analyze the provided \
image and extract architectural information.

Your analysis should identify:
1. Type of drawing (floor plan, elevation, section, detail)
2. Room layouts and dimensions
3. Structural elements (walls, columns, beams)
4. Doors, windows, and openings
5. Annotations and symbols
6. Scale and orientation

Respond in JSON format with:
{
    "summary": "Brief description of the drawing",
    "details": ["List of specific observations"],
    "confidence": 0.0-1.0 confidence score
}""",
    user_prompt_template="Analyze this architectural drawing and identify key elements.",
    output_format="json",
)

CIRCUIT_DIAGRAM_PROMPT = PromptTemplate(
    name="circuit_diagram",
    description="Analyze electrical circuit diagrams",
    system_prompt="""You are an expert electrical engineer. Analyze the provided circuit \
diagram and extract technical information.

Your analysis should identify:
1. Type of circuit (analog, digital, power, control)
2. Components (resistors, capacitors, ICs, etc.)
3. Component values and ratings
4. Connection topology
5. Input/output ports
6. Power supply requirements

Respond in JSON format with:
{
    "summary": "Brief description of the circuit",
    "details": ["List of specific observations"],
    "confidence": 0.0-1.0 confidence score
}""",
    user_prompt_template="Analyze this circuit diagram and identify components and topology.",
    output_format="json",
)

FLOWCHART_PROMPT = PromptTemplate(
    name="flowchart",
    description="Analyze flowcharts and process diagrams",
    system_prompt="""You are an expert process analyst. Analyze the provided flowchart or \
process diagram.

Your analysis should identify:
1. Type of diagram (process flow, data flow, decision tree)
2. Start and end points
3. Process steps and decisions
4. Data inputs and outputs
5. Loops and branches
6. Swimlanes or responsibility areas

Respond in JSON format with:
{
    "summary": "Brief description of the process",
    "details": ["List of specific observations"],
    "confidence": 0.0-1.0 confidence score
}""",
    user_prompt_template="Analyze this flowchart and describe the process flow.",
    output_format="json",
)

GENERAL_PROMPT = PromptTemplate(
    name="general",
    description="General image analysis",
    system_prompt="""You are an expert image analyst. Analyze the provided image and \
describe what you see.

Provide a comprehensive analysis including:
1. Main subject matter
2. Key elements and features
3. Text or annotations if present
4. Context and purpose
5. Quality and clarity assessment

Respond in JSON format with:
{
    "summary": "Brief description of the image",
    "details": ["List of specific observations"],
    "confidence": 0.0-1.0 confidence score
}""",
    user_prompt_template="Analyze this image and describe its contents.",
    output_format="json",
)

# Registry of built-in templates
BUILTIN_TEMPLATES: Dict[str, PromptTemplate] = {
    "engineering_drawing": ENGINEERING_DRAWING_PROMPT,
    "architectural": ARCHITECTURAL_PROMPT,
    "circuit_diagram": CIRCUIT_DIAGRAM_PROMPT,
    "flowchart": FLOWCHART_PROMPT,
    "general": GENERAL_PROMPT,
}


@dataclass
class PromptConfig:
    """Configuration for custom prompts."""

    template_type: PromptType = PromptType.ENGINEERING_DRAWING
    custom_system_prompt: Optional[str] = None
    custom_user_prompt: Optional[str] = None
    output_format: str = "json"
    additional_instructions: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    def get_prompts(self) -> tuple[str, str]:
        """
        Get the system and user prompts based on configuration.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if self.template_type == PromptType.CUSTOM:
            system = self.custom_system_prompt or GENERAL_PROMPT.system_prompt
            user = self.custom_user_prompt or GENERAL_PROMPT.user_prompt_template
        else:
            template = BUILTIN_TEMPLATES.get(
                self.template_type.value,
                GENERAL_PROMPT,
            )
            system, user = template.render(**self.variables)

        # Add additional instructions if provided
        if self.additional_instructions:
            system = f"{system}\n\nAdditional instructions:\n{self.additional_instructions}"

        return system, user


class PromptManager:
    """
    Manages prompt templates and configurations.

    Features:
    - Register custom templates
    - Template inheritance
    - Variable substitution
    - Prompt composition
    """

    def __init__(self) -> None:
        """Initialize prompt manager with built-in templates."""
        self._templates: Dict[str, PromptTemplate] = dict(BUILTIN_TEMPLATES)
        self._default_template = "engineering_drawing"

    def register_template(
        self,
        template: PromptTemplate,
        overwrite: bool = False,
    ) -> None:
        """
        Register a custom template.

        Args:
            template: The template to register
            overwrite: Whether to overwrite existing template

        Raises:
            ValueError: If template exists and overwrite=False
        """
        if template.name in self._templates and not overwrite:
            raise ValueError(
                f"Template '{template.name}' already exists. "
                "Use overwrite=True to replace."
            )
        self._templates[template.name] = template
        logger.info(f"Registered prompt template: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None if not found
        """
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template names
        """
        return list(self._templates.keys())

    def set_default(self, name: str) -> None:
        """
        Set the default template.

        Args:
            name: Template name to set as default

        Raises:
            ValueError: If template doesn't exist
        """
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found")
        self._default_template = name

    def get_prompts(
        self,
        template_name: Optional[str] = None,
        config: Optional[PromptConfig] = None,
        **variables: Any,
    ) -> tuple[str, str]:
        """
        Get prompts for analysis.

        Args:
            template_name: Optional template name (uses default if None)
            config: Optional PromptConfig (overrides template_name)
            **variables: Variables to substitute in template

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if config:
            return config.get_prompts()

        name = template_name or self._default_template
        template = self._templates.get(name, GENERAL_PROMPT)
        return template.render(**variables)

    def create_config(
        self,
        template_type: str = "engineering_drawing",
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        **variables: Any,
    ) -> PromptConfig:
        """
        Create a prompt configuration.

        Args:
            template_type: Type of template to use
            custom_system_prompt: Override system prompt
            custom_user_prompt: Override user prompt
            additional_instructions: Extra instructions to append
            **variables: Variables for template

        Returns:
            PromptConfig instance
        """
        try:
            prompt_type = PromptType(template_type)
        except ValueError:
            prompt_type = PromptType.CUSTOM

        return PromptConfig(
            template_type=prompt_type,
            custom_system_prompt=custom_system_prompt,
            custom_user_prompt=custom_user_prompt,
            additional_instructions=additional_instructions,
            variables=dict(variables),
        )


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance.

    Returns:
        PromptManager singleton
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompts(
    template_name: Optional[str] = None,
    **variables: Any,
) -> tuple[str, str]:
    """
    Convenience function to get prompts.

    Args:
        template_name: Optional template name
        **variables: Variables to substitute

    Returns:
        Tuple of (system_prompt, user_prompt)

    Example:
        >>> system, user = get_prompts("circuit_diagram")
        >>> print(system[:50])
        'You are an expert electrical engineer...'
    """
    manager = get_prompt_manager()
    return manager.get_prompts(template_name, **variables)


def register_custom_template(
    name: str,
    system_prompt: str,
    user_prompt: str,
    description: str = "",
    variables: Optional[List[str]] = None,
) -> PromptTemplate:
    """
    Register a custom prompt template.

    Args:
        name: Template name
        system_prompt: System prompt text
        user_prompt: User prompt template
        description: Template description
        variables: List of variable names in user_prompt

    Returns:
        The registered PromptTemplate

    Example:
        >>> template = register_custom_template(
        ...     name="pcb_layout",
        ...     system_prompt="You are a PCB design expert...",
        ...     user_prompt="Analyze this PCB layout for {focus_area}",
        ...     variables=["focus_area"],
        ... )
    """
    template = PromptTemplate(
        name=name,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
        description=description,
        variables=variables or [],
    )
    manager = get_prompt_manager()
    manager.register_template(template)
    return template
