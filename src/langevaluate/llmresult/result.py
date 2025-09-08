from dataclasses import dataclass, asdict
from typing import Union, List
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint
from rich.syntax import Syntax

@dataclass
class LLMResult:
    """평가 결과를 저장하기 위한 데이터 클래스"""
    input : str 
    output : str = None
    scoring_model_output : str = None
    expected_output : str = None
    context : Union[str, List[str]] = None
    retrieval_context : Union[str, List[str]] = None
    choices : List[str] = None
    score : float = None
    additional_info : dict = None
    metadata : dict = None
    
    
    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환"""
        return asdict(self)
    
        
    @classmethod
    def from_dict(cls, data: dict) -> 'LLMResult':
        """딕셔너리로부터 LLMResult 객체를 생성
        
        Args:
            data (dict): LLMResult 객체로 변환할 딕셔너리
            
        Returns:
            LLMResult: 생성된 LLMResult 객체
        """
        return cls(
            input=data["input"],
            output=data["output"],
            scoring_model_output=data["scoring_model_output"], 
            expected_output=data["expected_output"],
            context=data["context"],
            retrieval_context=data["retrieval_context"],
            choices=data["choices"],
            score=data["score"],
            additional_info=data['additional_info'],
            metadata=data["metadata"]
        )

    def pprint(self, max_length=1000):
        """Rich 라이브러리를 사용하여 LLMResult 객체를 예쁘게 출력합니다."""
        console = Console()
        
        # 전체 결과를 Panel로 감싸기
        main_tree = Tree("📊 [bold blue]LLMResult[/bold blue]")
        
        # 기본 필드 추가
        main_tree.add(f"🔍 [bold cyan]Input:[/bold cyan] {self._truncate_text(self.input, max_length)}")
        
        if self.output:
            output_syntax = Syntax(self._truncate_text(self.output, max_length), "markdown", theme="monokai", word_wrap=True)
            output_branch = main_tree.add("📝 [bold green]Output:[/bold green]")
            output_branch.add(output_syntax)
        
        if self.score is not None:
            score_icon = "🌟" if self.score > 0 else "⚠️" if self.score < 0 else "⭐"
            main_tree.add(f"{score_icon} [bold yellow]Score:[/bold yellow] {self.score}")
        
        # 추가 정보가 있는 경우
        if self.additional_info:
            add_info_branch = main_tree.add("📋 [bold magenta]Additional Info:[/bold magenta]")
            for key, value in self.additional_info.items():
                if value is not None:
                    if isinstance(value, str) and len(value) > 100:
                        add_info_branch.add(f"[bold]{key}:[/bold] {self._truncate_text(value, max_length)}")
                    else:
                        add_info_branch.add(f"[bold]{key}:[/bold] {value}")
        
        # 메타데이터가 있는 경우
        if self.metadata:
            meta_branch = main_tree.add("🔖 [bold cyan]Metadata:[/bold cyan]")
            for key, value in self.metadata.items():
                if value is not None:
                    if isinstance(value, dict):
                        meta_sub_branch = meta_branch.add(f"[bold]{key}:[/bold]")
                        for k, v in value.items():
                            meta_sub_branch.add(f"[italic]{k}:[/italic] {v}")
                    else:
                        meta_branch.add(f"[bold]{key}:[/bold] {value}")
        
        # 선택적 필드 추가
        self._add_optional_field(main_tree, "expected_output", "Expected Output", max_length, "🎯", "blue")
        self._add_optional_field(main_tree, "scoring_model_output", "Scoring Model Output", max_length, "⚖️", "purple")
        
        # 리스트 형태 필드 처리
        self._add_list_field(main_tree, "context", "Context", max_length, "📚", "cyan")
        self._add_list_field(main_tree, "retrieval_context", "Retrieval Context", max_length, "🔎", "green")
        self._add_list_field(main_tree, "choices", "Choices", max_length, "🔢", "yellow")
        
        # 패널로 감싸서 출력
        panel = Panel(main_tree, title="[bold]LLM Evaluation Result[/bold]", border_style="blue")
        console.print(panel)
    
    def _truncate_text(self, text, max_length):
        """텍스트가 너무 길 경우 잘라냅니다."""
        if not text:
            return ""
        if len(text) > max_length:
            return text[:max_length] + " [italic](truncated...)[/italic]"
        return text
    
    def _add_optional_field(self, tree, field_name, display_name, max_length, icon, color):
        """선택적 필드를 트리에 추가합니다."""
        value = getattr(self, field_name)
        if value:
            field_syntax = Syntax(self._truncate_text(value, max_length), "markdown", theme="monokai", word_wrap=True)
            field_branch = tree.add(f"{icon} [bold {color}]{display_name}:[/bold {color}]")
            field_branch.add(field_syntax)
    
    def _add_list_field(self, tree, field_name, display_name, max_length, icon, color):
        """리스트 형태의 필드를 트리에 추가합니다."""
        value = getattr(self, field_name)
        if value:
            field_branch = tree.add(f"{icon} [bold {color}]{display_name}:[/bold {color}]")
            if isinstance(value, list):
                for i, item in enumerate(value):
                    field_branch.add(f"[bold]Item {i+1}:[/bold] {self._truncate_text(str(item), max_length)}")
            else:
                field_branch.add(self._truncate_text(str(value), max_length))
