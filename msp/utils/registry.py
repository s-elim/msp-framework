import logging
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Type

# Initialize module-level logger
logger = logging.getLogger(__name__)


class Registry:
    """
    A registry to map strings to classes or functions.

    This class enables dynamic instantiation of modules based on configuration files.
    It acts as a centralized dictionary that maps string identifiers to Python objects
    (typically classes).

    Example:
        >>> BACKBONE_REGISTRY = Registry("BACKBONE")
        >>> @BACKBONE_REGISTRY.register()
        ... class ResNet:
        ...     pass
        >>> backbone_cls = BACKBONE_REGISTRY.get("ResNet")
    """

    def __init__(self, name: str) -> None:
        """
        Initializes a Registry instance.

        Args:
            name (str): The name of the registry (e.g., "BACKBONE", "DATASET").
                        Used primarily for informative error messages and logging.
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """str: The name of the registry."""
        return self._name

    def _do_register(self, name: str, obj: Any) -> None:
        """
        Internal method to register an object under a specific name.

        Args:
            name (str): The string identifier for the object.
            obj (Any): The object (class or function) to register.

        Raises:
            ValueError: If the name is already registered to a different object.
        """
        assert (
            name is not None
        ), f"Cannot register an object with a None name in registry '{self._name}'."

        if name in self._obj_map:
            # Prevent silent overwriting of registered components
            raise ValueError(
                f"An object named '{name}' is already registered in registry "
                f"'{self._name}'. Existing object: {self._obj_map[name]}."
            )
        
        self._obj_map[name] = obj
        logger.debug(f"Registered '{name}' in registry '{self._name}'.")

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        """
        Decorator to register a function or class.

        Args:
            name (Optional[str]): An optional custom name to register the object under.
                                  If None, the object's __name__ attribute is used.

        Returns:
            Callable[[Any], Any]: The decorator function.
        """

        def deco(func_or_class: Any) -> Any:
            # Use the provided name, or fallback to the class/function's __name__
            reg_name = name if name is not None else func_or_class.__name__
            self._do_register(reg_name, func_or_class)
            return func_or_class

        return deco

    def get(self, name: str) -> Any:
        """
        Retrieves a registered object by its string name.

        Args:
            name (str): The string identifier of the object.

        Returns:
            Any: The registered object (typically a class).

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._obj_map:
            registered_names = ", ".join(self._obj_map.keys())
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry. "
                f"Available objects: [{registered_names}]."
            )
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        """Checks if a name is registered."""
        return name in self._obj_map

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterates over registered (name, object) pairs."""
        return iter(self._obj_map.items())

    def keys(self) -> Iterable[str]:
        """Returns all registered names."""
        return self._obj_map.keys()

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self.keys())})"
