"""
Base classes for agents and tools in the AutoShort pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic, Callable

T = TypeVar('T')
U = TypeVar('U')

class Agent(ABC, Generic[T, U]):
    """Base class for all agents in the pipeline.
    
    Args:
        name: The name of the agent
        model: The model identifier used by this agent
        
    Attributes:
        name: The agent's name
        model: The model identifier
    """
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    @abstractmethod
    async def execute(self, input_data: T) -> U:
        """Execute the agent's task.
        
        Args:
            input_data: Input data for the agent's task
            
        Returns:
            The result of the agent's task
            
        Raises:
            Exception: If the agent fails to execute its task
        """
        pass

class Tool(ABC, Generic[T, U]):
    """Base class for all tools in the pipeline.
    
    Args:
        name: The name of the tool
        
    Attributes:
        name: The tool's name
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def use(self, input_data: T) -> U:
        """Use the tool to perform a task.
        
        Args:
            input_data: Input data for the tool's task
            
        Returns:
            The result of using the tool
            
        Raises:
            Exception: If the tool fails to perform its task
        """
        pass

class VoiceModule(ABC):
    """Base class for text-to-speech modules.
    
    This class defines the interface for text-to-speech functionality
    in the pipeline.
    """
    def __init__(self):
        pass

    @abstractmethod
    def generate_voice(self, text: str, output_file: str) -> str:
        """Generate speech from text and save to a file.
        
        Args:
            text: The text to convert to speech
            output_file: Path where the audio file should be saved
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If speech generation fails
        """
        pass

class Node:
    """A node in the processing graph.
    
    Args:
        agent: Optional agent associated with this node
        tool: Optional tool associated with this node
        
    Attributes:
        agent: The agent instance if any
        tool: The tool instance if any
        edges: List of outgoing edges from this node
    """
    def __init__(
        self, 
        agent: Optional[Agent] = None, 
        tool: Optional[Tool] = None
    ):
        self.agent = agent
        self.tool = tool
        self.edges: list['Edge'] = []

    async def process(self, input_data: Any) -> Any:
        """Process input data through this node.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processed data
            
        Raises:
            ValueError: If node has neither agent nor tool
        """
        if self.agent:
            return await self.agent.execute(input_data)
        elif self.tool:
            return await self.tool.use(input_data)
        else:
            raise ValueError("Node has neither agent nor tool")

class Edge:
    """An edge in the processing graph.
    
    Args:
        source: Source node
        target: Target node
        condition: Optional condition function for edge traversal
        
    Attributes:
        source: The source node
        target: The target node
        condition: Optional condition function
    """
    def __init__(
        self, 
        source: Node, 
        target: Node, 
        condition: Optional[Callable[[Any], bool]] = None
    ):
        self.source = source
        self.target = target
        self.condition = condition

class Graph:
    """A graph representing the processing pipeline.
    
    Attributes:
        nodes: List of nodes in the graph
        edges: List of edges in the graph
    """
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        Args:
            node: The node to add
        """
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: The edge to add
        """
        self.edges.append(edge)
        edge.source.edges.append(edge) 