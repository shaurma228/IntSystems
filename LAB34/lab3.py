from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Type, Set, Callable, TypeVar, cast


# Факт — это отдельная "порция знания" экспертной системы.
# Например: выбран процессор, известен блок питания, проверена совместимость.
class Fact:
    """Базовый класс фактов."""


# TFact нужен только для подсказок типов в get_facts:
# он означает "любой конкретный класс, который является Fact".
TFact = TypeVar("TFact", bound=Fact)


# dataclass автоматически создает __init__, сравнение объектов и красивый вывод.
# frozen=True запрещает менять объект после создания и позволяет хранить его в set.
@dataclass(frozen=True)
class UserRequirement(Fact):
    # Требования пользователя к будущей сборке.
    purpose: str
    budget: int
    resolution: str


@dataclass(frozen=True)
class CPU(Fact):
    # Информация о процессоре.
    model: str
    socket: str
    tdp: int


@dataclass(frozen=True)
class Motherboard(Fact):
    # Информация о материнской плате.
    model: str
    socket: str
    ram_type: str


@dataclass(frozen=True)
class GPU(Fact):
    # Информация о видеокарте.
    model: str
    tdp: int


@dataclass(frozen=True)
class PSU(Fact):
    # Информация о блоке питания.
    power: int


@dataclass(frozen=True)
class Compatibility(Fact):
    # Результат проверки совместимости CPU и материнской платы.
    status: str


@dataclass(frozen=True)
class PowerStatus(Fact):
    # Результат проверки мощности блока питания.
    status: str


@dataclass(frozen=True)
class BuildResult(Fact):
    # Финальный результат проверки сборки.
    status: str


@dataclass(frozen=True)
class SelectedComponent(Fact):
    # Текстовое сообщение для вывода промежуточных результатов.
    description: str


# Rule — декоратор, который помечает метод как правило экспертной системы.
# После такой пометки Engine сможет автоматически найти и запустить это правило.
def Rule(name: str):
    def decorator(func):
        # Эти поля добавляются прямо к функции.
        # Потом _collect_rules ищет функции с _is_rule == True.
        func._is_rule = True
        func._rule_name = name
        return func

    return decorator


class Engine:
    def __init__(self):
        # facts хранит все известные факты.
        # Ключ словаря — тип факта, например CPU.
        # Значение — множество фактов этого типа.
        self.facts: Dict[Type[Fact], Set[Fact]] = {}

        # Сюда попадут методы, помеченные декоратором @Rule.
        self.rules: List[Callable[[], None]] = []

        # Здесь хранятся имена правил, которые реально добавили новые факты.
        self.fired_rules: List[str] = []
        self._collect_rules()

    def _collect_rules(self):
        # dir(self) возвращает имена всех полей и методов объекта.
        # Мы перебираем их и ищем методы, которые были помечены как правила.
        for attr in dir(self):
            method = getattr(self, attr)
            if callable(method) and getattr(method, "_is_rule", False):
                self.rules.append(method)

    def assert_fact(self, fact: Fact):
        # Метод добавляет новый факт в базу знаний.
        cls = type(fact)

        # Если фактов такого типа еще не было, создаем для них пустое множество.
        if cls not in self.facts:
            self.facts[cls] = set()

        # Один и тот же факт не добавляется дважды.
        if fact not in self.facts[cls]:
            self.facts[cls].add(fact)
            return True
        return False

    def get_facts(self, fact_type: Type[TFact]) -> List[TFact]:
        # dict хранит Set[Fact], поэтому приводим тип явно
        return cast(List[TFact], list(self.facts.get(fact_type, set())))

    def run(self):
        # Правила могут добавлять новые факты.
        # Новые факты могут позволить сработать другим правилам.
        # Поэтому запускаем правила по кругу, пока появляются изменения.
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                # Считаем количество фактов до запуска правила.
                before = sum(len(v) for v in self.facts.values())
                rule()
                # Считаем количество фактов после запуска правила.
                after = sum(len(v) for v in self.facts.values())
                if after > before:
                    # Если фактов стало больше, значит правило сработало.
                    rule_name = getattr(rule, "_rule_name", rule.__name__)
                    if rule_name not in self.fired_rules:
                        self.fired_rules.append(rule_name)
                    changed = True


class PCExpertSystem(Engine):

    @Rule("R1_CPU_for_Gaming")
    def select_cpu(self):
        # Если пользователь собирает игровой ПК и бюджет достаточно большой,
        # система выбирает подходящий процессор.
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры" and req.budget > 60000:
                cpu = CPU("Ryzen 5 5600", "AM4", 65)
                self.assert_fact(cpu)
                self.assert_fact(SelectedComponent(f"Выбран CPU: {cpu.model}"))

    @Rule("R5_GPU_for_Gaming")
    def select_gpu(self):
        # Для игровой сборки под Full HD выбираем видеокарту RTX 3060.
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

        # CPU и материнская плата совместимы, если у них одинаковый socket.
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
        # Если socket у CPU и материнской платы разный, сборка несовместима.
        for cpu in self.get_facts(CPU):
            for mb in self.get_facts(Motherboard):
                if cpu.socket != mb.socket:
                    if Compatibility in self.facts:
                        self.facts[Compatibility].clear()
                    self.assert_fact(Compatibility("несовместимы"))
                    self.assert_fact(SelectedComponent("CPU и MB несовместимы"))

    @Rule("R6_PSU_Sufficient")
    def check_power(self):
        # Считаем примерное энергопотребление CPU и GPU.
        total_tdp = sum(cpu.tdp for cpu in self.get_facts(CPU)) + sum(gpu.tdp for gpu in self.get_facts(GPU))

        # Добавляем запас 20%, чтобы блок питания не работал "впритык".
        total_tdp = int(total_tdp * 1.2)

        # Старый статус мощности удаляем, чтобы не было двух разных ответов сразу.
        if PowerStatus in self.facts:
            self.facts[PowerStatus].clear()

        # Старое сообщение про БП тоже удаляем перед добавлением нового.
        if SelectedComponent in self.facts:
            selected = cast(Set[SelectedComponent], self.facts[SelectedComponent])
            self.facts[SelectedComponent] = {f for f in selected if "БП" not in f.description}

        # Проверяем каждый известный блок питания.
        for psu in self.get_facts(PSU):
            if psu.power >= total_tdp:
                self.assert_fact(PowerStatus("достаточно"))
                self.assert_fact(SelectedComponent(f"Мощность БП достаточна: {psu.power}W"))
            else:
                self.assert_fact(PowerStatus("недостаточно"))
                self.assert_fact(SelectedComponent(f"Мощность БП недостаточна: {psu.power}W"))

    @Rule("R_Final_Build")
    def final_build(self):
        # Финальный вывод можно делать только после двух проверок:
        # совместимости CPU/MB и достаточности блока питания.
        comp = self.get_facts(Compatibility)
        power = self.get_facts(PowerStatus)
        if comp and power:
            if comp[0].status == "совместимы" and power[0].status == "достаточно":
                self.assert_fact(BuildResult("Сборка корректна"))
            else:
                self.assert_fact(BuildResult("Требуется замена компонентов"))


def interactive_input():
    # Ручной ввод исходных фактов от пользователя.
    print("\n=== Ввод исходных данных для сборки ПК ===")
    purpose = input("Цель сборки (игры/офис/другое): ").strip()

    # Бюджет должен быть числом, поэтому ввод повторяется до корректного значения.
    while True:
        try:
            budget = int(input("Бюджет (руб): ").strip())
            break
        except ValueError:
            print("Ошибка: бюджет должен быть целым числом.")
    resolution = input("Разрешение монитора (1080p/1440p/4k): ").strip()
    mb_socket = input("Сокет материнской платы (AM4/LGA1200/...): ").strip()

    # Мощность БП тоже должна быть числом.
    while True:
        try:
            psu_power = int(input("Мощность БП (Вт): ").strip())
            break
        except ValueError:
            print("Ошибка: мощность БП должна быть целым числом.")
    return purpose, budget, resolution, mb_socket, psu_power


def run_scenario(name, motherboard_socket, psu_power, purpose="игры", budget=80000, resolution="1080p"):
    # Один сценарий — это один запуск экспертной системы с конкретными входными данными.
    print("\n==============================")
    print(f"Сценарий: {name}")
    print("==============================")

    engine = PCExpertSystem()

    # Добавляем начальные факты, которые система знает до запуска правил.
    engine.assert_fact(UserRequirement(purpose, budget, resolution))
    engine.assert_fact(Motherboard("MSI B550", motherboard_socket, "DDR4"))
    engine.assert_fact(PSU(psu_power))

    # После запуска движок сам применит все подходящие правила.
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
    # Пользователь выбирает: ввести свои данные или запустить готовые примеры.
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
