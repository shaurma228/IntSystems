from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Type, TypeVar, cast


class Fact:
    """Базовый класс фактов."""


TFact = TypeVar("TFact", bound=Fact)


@dataclass(frozen=True)
class UserRequirement(Fact):
    purpose: str
    budget: int
    resolution: str


@dataclass(frozen=True)
class CPU(Fact):
    model: str
    socket: str
    tdp: int


@dataclass(frozen=True)
class Motherboard(Fact):
    model: str
    socket: str
    ram_type: str


@dataclass(frozen=True)
class GPU(Fact):
    model: str
    tdp: int


@dataclass(frozen=True)
class PSU(Fact):
    power: int


@dataclass(frozen=True)
class Compatibility(Fact):
    status: str


@dataclass(frozen=True)
class PowerStatus(Fact):
    status: str


@dataclass(frozen=True)
class BuildResult(Fact):
    status: str


@dataclass(frozen=True)
class SelectedComponent(Fact):
    description: str


def Rule(name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        func._is_rule = True
        func._rule_name = name
        return func

    return decorator


class Engine:
    def __init__(self) -> None:
        self.facts: Dict[Type[Fact], Set[Fact]] = {}
        self.rules: List[Callable[[], None]] = []
        self.fired_rules: List[str] = []
        self._collect_rules()

    def _collect_rules(self) -> None:
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            if callable(method) and getattr(method, "_is_rule", False):
                self.rules.append(method)

    def assert_fact(self, fact: Fact) -> bool:
        fact_type = type(fact)
        self.facts.setdefault(fact_type, set())
        if fact in self.facts[fact_type]:
            return False
        self.facts[fact_type].add(fact)
        return True

    def get_facts(self, fact_type: Type[TFact]) -> List[TFact]:
        return cast(List[TFact], list(self.facts.get(fact_type, set())))

    def clear_facts(self, fact_type: Type[Fact]) -> None:
        if fact_type in self.facts:
            self.facts[fact_type].clear()

    def total_facts(self) -> int:
        return sum(len(items) for items in self.facts.values())

    def run(self) -> None:
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                before = self.total_facts()
                rule()
                after = self.total_facts()
                if after > before:
                    rule_name = getattr(rule, "_rule_name", rule.__name__)
                    if rule_name not in self.fired_rules:
                        self.fired_rules.append(rule_name)
                    changed = True


class PCExpertSystem(Engine):
    GAMING_CPU = CPU("Ryzen 5 5600", "AM4", 65)
    GAMING_GPU = GPU("RTX 3060", 170)

    def _replace_single_status(self, fact: Fact) -> None:
        self.clear_facts(type(fact))
        self.assert_fact(fact)

    def _replace_psu_message(self, message: str) -> None:
        selected = cast(Set[SelectedComponent], self.facts.get(SelectedComponent, set()))
        self.facts[SelectedComponent] = {fact for fact in selected if "БП" not in fact.description}
        self.assert_fact(SelectedComponent(message))

    @Rule("R1_CPU_for_Gaming")
    def select_cpu(self) -> None:
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры" and req.budget > 60000:
                self.assert_fact(self.GAMING_CPU)
                self.assert_fact(SelectedComponent(f"Выбран CPU: {self.GAMING_CPU.model}"))

    @Rule("R5_GPU_for_Gaming")
    def select_gpu(self) -> None:
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры" and req.resolution == "1080p":
                self.assert_fact(self.GAMING_GPU)
                self.assert_fact(SelectedComponent(f"Выбрана GPU: {self.GAMING_GPU.model}"))

    @Rule("R3_CPU_MB_Compatibility")
    def check_socket(self) -> None:
        if any(item.status == "несовместимы" for item in self.get_facts(Compatibility)):
            return

        for cpu in self.get_facts(CPU):
            for motherboard in self.get_facts(Motherboard):
                if cpu.socket == motherboard.socket:
                    self._replace_single_status(Compatibility("совместимы"))
                    self.assert_fact(SelectedComponent("CPU и MB совместимы"))

    @Rule("R4_CPU_MB_Not_Compatible")
    def check_socket_fail(self) -> None:
        for cpu in self.get_facts(CPU):
            for motherboard in self.get_facts(Motherboard):
                if cpu.socket != motherboard.socket:
                    self._replace_single_status(Compatibility("несовместимы"))
                    self.assert_fact(SelectedComponent("CPU и MB несовместимы"))

    @Rule("R6_PSU_Sufficient")
    def check_power(self) -> None:
        total_tdp = sum(cpu.tdp for cpu in self.get_facts(CPU))
        total_tdp += sum(gpu.tdp for gpu in self.get_facts(GPU))
        required_power = int(total_tdp * 1.2)

        self.clear_facts(PowerStatus)
        for psu in self.get_facts(PSU):
            if psu.power >= required_power:
                self.assert_fact(PowerStatus("достаточно"))
                self._replace_psu_message(f"Мощность БП достаточна: {psu.power}W")
            else:
                self.assert_fact(PowerStatus("недостаточно"))
                self._replace_psu_message(f"Мощность БП недостаточна: {psu.power}W")

    @Rule("R_Final_Build")
    def final_build(self) -> None:
        compatibility = self.get_facts(Compatibility)
        power = self.get_facts(PowerStatus)
        if compatibility and power:
            if compatibility[0].status == "совместимы" and power[0].status == "достаточно":
                self.assert_fact(BuildResult("Сборка корректна"))
            else:
                self.assert_fact(BuildResult("Требуется замена компонентов"))


def read_int(prompt: str, error_message: str) -> int:
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print(error_message)


def interactive_input() -> tuple[str, int, str, str, int]:
    print("\n=== Ввод исходных данных для сборки ПК ===")
    purpose = input("Цель сборки (игры/офис/другое): ").strip()
    budget = read_int("Бюджет (руб): ", "Ошибка: бюджет должен быть целым числом.")
    resolution = input("Разрешение монитора (1080p/1440p/4k): ").strip()
    motherboard_socket = input("Сокет материнской платы (AM4/LGA1200/...): ").strip()
    psu_power = read_int("Мощность БП (Вт): ", "Ошибка: мощность БП должна быть целым числом.")
    return purpose, budget, resolution, motherboard_socket, psu_power


def print_selected_components(engine: PCExpertSystem) -> None:
    print("\nПромежуточные факты:")
    for fact in sorted(engine.get_facts(SelectedComponent), key=lambda item: item.description):
        print(f"- {fact.description}")


def print_build_result(engine: PCExpertSystem) -> None:
    print("\nРезультат сборки:")
    for result in engine.get_facts(BuildResult):
        print(f"- {result.status}")


def print_fired_rules(engine: PCExpertSystem) -> None:
    print("\nСработавшие правила:")
    for rule in engine.fired_rules:
        print(f"- {rule}")


def run_scenario(
    name: str,
    motherboard_socket: str,
    psu_power: int,
    purpose: str = "игры",
    budget: int = 80000,
    resolution: str = "1080p",
) -> None:
    print("\n==============================")
    print(f"Сценарий: {name}")
    print("==============================")

    engine = PCExpertSystem()
    engine.assert_fact(UserRequirement(purpose, budget, resolution))
    engine.assert_fact(Motherboard("MSI B550", motherboard_socket, "DDR4"))
    engine.assert_fact(PSU(psu_power))
    engine.run()

    print_selected_components(engine)
    print_build_result(engine)
    print_fired_rules(engine)


def main() -> None:
    choice = input("Вы хотите ввести данные вручную? (y/n): ").strip().lower()
    if choice == "y":
        purpose, budget, resolution, motherboard_socket, psu_power = interactive_input()
        run_scenario("Интерактивная сборка", motherboard_socket, psu_power, purpose, budget, resolution)
        return

    run_scenario("Корректная сборка", "AM4", 600)
    run_scenario("Несовместимая материнская плата", "LGA1200", 600)


if __name__ == "__main__":
    main()
