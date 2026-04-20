from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Type, Set, Callable, TypeVar, cast


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


def Rule(name: str):
    def decorator(func):
        func._is_rule = True
        func._rule_name = name
        return func

    return decorator


class Engine:
    def __init__(self):
        self.facts: Dict[Type[Fact], Set[Fact]] = {}
        self.rules: List[Callable[[], None]] = []
        self.fired_rules: List[str] = []
        self._collect_rules()

    def _collect_rules(self):
        for attr in dir(self):
            method = getattr(self, attr)
            if callable(method) and getattr(method, "_is_rule", False):
                self.rules.append(method)

    def assert_fact(self, fact: Fact):
        cls = type(fact)
        if cls not in self.facts:
            self.facts[cls] = set()
        if fact not in self.facts[cls]:
            self.facts[cls].add(fact)
            return True
        return False

    def get_facts(self, fact_type: Type[TFact]) -> List[TFact]:
        # dict хранит Set[Fact], поэтому приводим тип явно
        return cast(List[TFact], list(self.facts.get(fact_type, set())))

    def run(self):
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                before = sum(len(v) for v in self.facts.values())
                rule()
                after = sum(len(v) for v in self.facts.values())
                if after > before:
                    rule_name = getattr(rule, "_rule_name", rule.__name__)
                    if rule_name not in self.fired_rules:
                        self.fired_rules.append(rule_name)
                    changed = True


class PCExpertSystem(Engine):

    @Rule("R1_CPU_for_Gaming")
    def select_cpu(self):
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры" and req.budget > 60000:
                cpu = CPU("Ryzen 5 5600", "AM4", 65)
                self.assert_fact(cpu)
                self.assert_fact(SelectedComponent(f"Выбран CPU: {cpu.model}"))

    @Rule("R5_GPU_for_Gaming")
    def select_gpu(self):
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры" and req.resolution == "1080p":
                gpu = GPU("RTX 3060", 170)
                self.assert_fact(gpu)
                self.assert_fact(SelectedComponent(f"Выбрана GPU: {gpu.model}"))

    @Rule("R3_CPU_MB_Compatibility")
    def check_socket(self):
        # Если уже установили "несовместимы" — не переопределяем.
        comp = self.get_facts(Compatibility)
        if any(c.status == "несовместимы" for c in comp):
            return
        for cpu in self.get_facts(CPU):
            for mb in self.get_facts(Motherboard):
                if cpu.socket == mb.socket:
                    # Делаем статус совместимости единственным.
                    if Compatibility in self.facts:
                        self.facts[Compatibility].clear()
                    self.assert_fact(Compatibility("совместимы"))
                    self.assert_fact(SelectedComponent("CPU и MB совместимы"))

    @Rule("R4_CPU_MB_Not_Compatible")
    def check_socket_fail(self):
        for cpu in self.get_facts(CPU):
            for mb in self.get_facts(Motherboard):
                if cpu.socket != mb.socket:
                    if Compatibility in self.facts:
                        self.facts[Compatibility].clear()
                    self.assert_fact(Compatibility("несовместимы"))
                    self.assert_fact(SelectedComponent("CPU и MB несовместимы"))

    @Rule("R6_PSU_Sufficient")
    def check_power(self):
        total_tdp = sum(cpu.tdp for cpu in self.get_facts(CPU)) + sum(gpu.tdp for gpu in self.get_facts(GPU))
        total_tdp = int(total_tdp * 1.2)

        if PowerStatus in self.facts:
            self.facts[PowerStatus].clear()
        if SelectedComponent in self.facts:
            selected = cast(Set[SelectedComponent], self.facts[SelectedComponent])
            self.facts[SelectedComponent] = {f for f in selected if "БП" not in f.description}

        for psu in self.get_facts(PSU):
            if psu.power >= total_tdp:
                self.assert_fact(PowerStatus("достаточно"))
                self.assert_fact(SelectedComponent(f"Мощность БП достаточна: {psu.power}W"))
            else:
                self.assert_fact(PowerStatus("недостаточно"))
                self.assert_fact(SelectedComponent(f"Мощность БП недостаточна: {psu.power}W"))

    @Rule("R_Final_Build")
    def final_build(self):
        comp = self.get_facts(Compatibility)
        power = self.get_facts(PowerStatus)
        if comp and power:
            if comp[0].status == "совместимы" and power[0].status == "достаточно":
                self.assert_fact(BuildResult("Сборка корректна"))
            else:
                self.assert_fact(BuildResult("Требуется замена компонентов"))


def interactive_input():
    print("\n=== Ввод исходных данных для сборки ПК ===")
    purpose = input("Цель сборки (игры/офис/другое): ").strip()
    while True:
        try:
            budget = int(input("Бюджет (руб): ").strip())
            break
        except ValueError:
            print("Ошибка: бюджет должен быть целым числом.")
    resolution = input("Разрешение монитора (1080p/1440p/4k): ").strip()
    mb_socket = input("Сокет материнской платы (AM4/LGA1200/...): ").strip()
    while True:
        try:
            psu_power = int(input("Мощность БП (Вт): ").strip())
            break
        except ValueError:
            print("Ошибка: мощность БП должна быть целым числом.")
    return purpose, budget, resolution, mb_socket, psu_power


def run_scenario(name, motherboard_socket, psu_power, purpose="игры", budget=80000, resolution="1080p"):
    print("\n==============================")
    print(f"Сценарий: {name}")
    print("==============================")

    engine = PCExpertSystem()
    engine.assert_fact(UserRequirement(purpose, budget, resolution))
    engine.assert_fact(Motherboard("MSI B550", motherboard_socket, "DDR4"))
    engine.assert_fact(PSU(psu_power))

    engine.run()

    print("\nПромежуточные факты:")
    for fact in sorted(engine.get_facts(SelectedComponent), key=lambda f: f.description):
        print(f"- {fact.description}")

    print("\nРезультат сборки:")
    for result in engine.get_facts(BuildResult):
        print(f"- {result.status}")

    print("\nСработавшие правила:")
    for rule in engine.fired_rules:
        print(f"- {rule}")


def main() -> None:
    choice = input("Вы хотите ввести данные вручную? (y/n): ").strip().lower()
    if choice == "y":
        purpose, budget, resolution, mb_socket, psu_power = interactive_input()
        run_scenario("Интерактивная сборка", mb_socket, psu_power, purpose, budget, resolution)
    else:
        # Hardcode-сценарии
        run_scenario("Корректная сборка", "AM4", 600)
        run_scenario("Несовместимая материнская плата", "LGA1200", 600)


if __name__ == "__main__":
    main()


