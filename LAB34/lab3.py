"""
Экспертная система для подбора и проверки совместимости комплектующих ПК.

Особенности:
- Проверка совместимости CPU и материнской платы по сокету.
- Расчёт необходимой мощности блока питания с учётом TDP компонентов,
  запаса на пиковые нагрузки и округления до стандартных значений (шаг 50 Вт).
- Автоматический выбор CPU/GPU в зависимости от бюджета (для демо-сценариев).
- Ручной ввод любых компонентов с полной проверкой.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Type, TypeVar, cast


# ---------- Базовые классы для фактов ----------
class Fact:
    """Базовый класс для всех фактов экспертной системы."""
    pass


TFact = TypeVar("TFact", bound=Fact)


@dataclass(frozen=True)
class UserRequirement(Fact):
    """Требования пользователя к сборке (автоматический режим)."""
    purpose: str       # цель: "игры", "работа" и т.д.
    budget: int        # бюджет в рублях
    resolution: str    # разрешение экрана ("1080p", "1440p" и т.д.)


@dataclass(frozen=True)
class CPU(Fact):
    """Факт о процессоре."""
    model: str         # модель CPU
    socket: str        # сокет (например, AM4, LGA1700)
    tdp: int           # тепловыделение в ваттах


@dataclass(frozen=True)
class Motherboard(Fact):
    """Факт о материнской плате."""
    model: str
    socket: str
    ram_type: str      # тип оперативной памяти (DDR4, DDR5 и т.д.)


@dataclass(frozen=True)
class GPU(Fact):
    """Факт о видеокарте."""
    model: str
    tdp: int


@dataclass(frozen=True)
class PSU(Fact):
    """Факт о блоке питания."""
    power: int         # мощность в ваттах


@dataclass(frozen=True)
class Compatibility(Fact):
    """Результат проверки совместимости CPU и материнской платы по сокету."""
    status: str        # "совместимы" или "несовместимы"


@dataclass(frozen=True)
class PowerStatus(Fact):
    """Результат проверки достаточности мощности БП."""
    status: str               # "достаточно" или "недостаточно"
    recommended_wattage: int  # рекомендуемая мощность (округлённая)


@dataclass(frozen=True)
class BuildResult(Fact):
    """Итоговый вердикт по сборке."""
    status: str        # "Сборка корректна" или "Требуется замена компонентов"


@dataclass(frozen=True)
class SelectedComponent(Fact):
    """Промежуточное сообщение о выбранном или проверенном компоненте."""
    description: str


# ---------- Декоратор для правил ----------
def Rule(name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Декоратор, помечающий метод как правило экспертной системы.
    Правила автоматически собираются движком и выполняются до насыщения фактами.
    """
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        func._is_rule = True      # метка, что это правило
        func._rule_name = name    # имя правила для вывода
        return func
    return decorator


# ---------- Движок экспертной системы ----------
class Engine:
    """Простой движок продукционной системы с прямым выводом."""

    def __init__(self) -> None:
        # Хранилище фактов: ключ – тип факта, значение – множество фактов данного типа
        self.facts: Dict[Type[Fact], Set[Fact]] = {}
        # Список правил (методов, помеченных @Rule)
        self.rules: List[Callable[[], None]] = []
        # Список имён уже сработавших правил (для отладки)
        self.fired_rules: List[str] = []
        self._collect_rules()

    def _collect_rules(self) -> None:
        """Собирает все методы текущего экземпляра, помеченные как правила."""
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            if callable(method) and getattr(method, "_is_rule", False):
                self.rules.append(method)

    def assert_fact(self, fact: Fact) -> bool:
        """
        Добавляет факт в рабочую память, если его там ещё нет.
        Возвращает True, если факт был добавлен, иначе False.
        """
        fact_type = type(fact)
        self.facts.setdefault(fact_type, set())
        if fact in self.facts[fact_type]:
            return False
        self.facts[fact_type].add(fact)
        return True

    def get_facts(self, fact_type: Type[TFact]) -> List[TFact]:
        """Возвращает список всех фактов указанного типа."""
        return list(self.facts.get(fact_type, set()))

    def clear_facts(self, fact_type: Type[Fact]) -> None:
        """Удаляет все факты заданного типа (очищает множество)."""
        if fact_type in self.facts:
            self.facts[fact_type].clear()

    def total_facts(self) -> int:
        """Общее количество фактов в системе."""
        return sum(len(items) for items in self.facts.values())

    def run(self) -> None:
        """
        Запускает выполнение правил.
        Цикл продолжается, пока хотя бы одно правило добавило новые факты.
        """
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                before = self.total_facts()
                rule()                      # выполнение правила
                after = self.total_facts()
                if after > before:
                    # Правило сработало и добавило факты
                    rule_name = getattr(rule, "_rule_name", rule.__name__)
                    if rule_name not in self.fired_rules:
                        self.fired_rules.append(rule_name)
                    changed = True


# ---------- Конкретная экспертная система для ПК ----------
class PCExpertSystem(Engine):
    """
    Экспертная система для проверки совместимости комплектующих ПК.
    Содержит правила выбора компонентов (автоматический режим),
    правила проверки сокета и расчёта мощности БП с округлением.
    """

    # ---- Предопределённые компоненты для автоматического подбора ----
    # CPU для разных бюджетов
    LOW_BUDGET_CPU = CPU("Ryzen 3 4100", "AM4", 65)
    MID_BUDGET_CPU = CPU("Ryzen 5 5600", "AM4", 65)
    HIGH_BUDGET_CPU = CPU("Ryzen 7 5800X", "AM4", 105)

    # GPU для разных бюджетов
    LOW_BUDGET_GPU = GPU("GTX 1650", 75)
    MID_BUDGET_GPU = GPU("RTX 3060", 170)
    HIGH_BUDGET_GPU = GPU("RTX 3070 Ti", 290)   # реальный TDP около 290 Вт

    # ---- Вспомогательные методы для управления фактами ----
    def _replace_single_status(self, fact: Fact) -> None:
        """
        Заменяет все факты того же типа, что и fact, одним новым fact.
        Используется для статусов, где должен быть только один актуальный результат.
        """
        self.clear_facts(type(fact))
        self.assert_fact(fact)

    def _replace_psu_message(self, message: str) -> None:
        """
        Обновляет сообщение о блоке питания, удаляя старые сообщения на тему БП,
        чтобы не накапливалось много дублирующих фактов.
        """
        selected = cast(Set[SelectedComponent], self.facts.get(SelectedComponent, set()))
        # Оставляем только сообщения, не содержащие "БП"
        self.facts[SelectedComponent] = {fact for fact in selected if "БП" not in fact.description}
        self.assert_fact(SelectedComponent(message))

    def _replace_cpu(self, cpu: CPU) -> None:
        """Заменяет ранее выбранный CPU новым (только один CPU в системе)."""
        self.clear_facts(CPU)
        self.assert_fact(cpu)

    def _replace_gpu(self, gpu: GPU) -> None:
        """Заменяет ранее выбранную GPU новой (только одна GPU в системе)."""
        self.clear_facts(GPU)
        self.assert_fact(gpu)

    @staticmethod
    def _round_psu_wattage(wattage: int) -> int:
        """
        Округляет расчётную мощность БП до реально существующих значений.
        Правило: минимум 450 Вт, далее округление вверх до ближайшего числа,
        кратного 50 (например, 437 → 450, 612 → 650, 600 → 600).
        """
        if wattage < 450:
            return 450
        remainder = wattage % 50
        if remainder == 0:
            return wattage
        return wattage + (50 - remainder)

    # ---- Правила выбора компонентов (только для автоматического режима) ----
    @Rule("R1_CPU_Selection")
    def select_cpu(self) -> None:
        """
        Правило выбора CPU в зависимости от бюджета и цели.
        Вызывается, если есть факт UserRequirement.
        Заменяет любой ранее выбранный CPU.
        """
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры":
                if req.budget < 50000:
                    self._replace_cpu(self.LOW_BUDGET_CPU)
                    self.assert_fact(SelectedComponent(f"Выбран CPU: {self.LOW_BUDGET_CPU.model}"))
                elif req.budget < 80000:
                    self._replace_cpu(self.MID_BUDGET_CPU)
                    self.assert_fact(SelectedComponent(f"Выбран CPU: {self.MID_BUDGET_CPU.model}"))
                else:
                    self._replace_cpu(self.HIGH_BUDGET_CPU)
                    self.assert_fact(SelectedComponent(f"Выбран CPU: {self.HIGH_BUDGET_CPU.model}"))

    @Rule("R2_GPU_Selection")
    def select_gpu(self) -> None:
        """
        Правило выбора GPU в зависимости от бюджета и разрешения.
        Для 1080p при низком бюджете берём более скромную видеокарту.
        """
        for req in self.get_facts(UserRequirement):
            if req.purpose.lower() == "игры":
                if req.budget < 50000 and req.resolution == "1080p":
                    self._replace_gpu(self.LOW_BUDGET_GPU)
                    self.assert_fact(SelectedComponent(f"Выбрана GPU: {self.LOW_BUDGET_GPU.model}"))
                elif req.budget < 80000:
                    self._replace_gpu(self.MID_BUDGET_GPU)
                    self.assert_fact(SelectedComponent(f"Выбрана GPU: {self.MID_BUDGET_GPU.model}"))
                else:
                    self._replace_gpu(self.HIGH_BUDGET_GPU)
                    self.assert_fact(SelectedComponent(f"Выбрана GPU: {self.HIGH_BUDGET_GPU.model}"))

    # ---- Правила проверки совместимости ----
    @Rule("R3_CPU_MB_Socket_Compatibility")
    def check_socket(self) -> None:
        """
        Проверяет совместимость сокетов CPU и материнской платы.
        Если уже есть статус "несовместимы", повторно не проверяем (оптимизация).
        Иначе перебираем все пары CPU и MB и выставляем соответствующий статус.
        """
        # Если уже зафиксирована несовместимость, не перезаписываем
        if any(item.status == "несовместимы" for item in self.get_facts(Compatibility)):
            return

        for cpu in self.get_facts(CPU):
            for motherboard in self.get_facts(Motherboard):
                if cpu.socket == motherboard.socket:
                    self._replace_single_status(Compatibility("совместимы"))
                    self.assert_fact(SelectedComponent(
                        f"CPU и MB совместимы (сокет): {cpu.model} ({cpu.socket}) <-> {motherboard.model} ({motherboard.socket})"
                    ))
                else:
                    # Если хотя бы одна пара несовместима, выставляем несовместимость
                    self._replace_single_status(Compatibility("несовместимы"))
                    self.assert_fact(SelectedComponent(
                        f"CPU и MB НЕсовместимы (сокет): {cpu.model} ({cpu.socket}) <-> {motherboard.model} ({motherboard.socket})"
                    ))

    @Rule("R4_PSU_Sufficient")
    def check_power(self) -> None:
        """
        Рассчитывает необходимую мощность блока питания.
        Формула: (сумма TDP CPU и GPU + 70 Вт на остальное) * коэффициент запаса.
        Коэффициент:
          - 1.5 для суммарного TDP ≤ 400 Вт
          - 1.75 для более мощных систем (из-за высоких пиковых нагрузок)
        Затем округляет до стандартного значения (шаг 50 Вт, минимум 450).
        Сравнивает с мощностью выбранного БП и выдаёт статус.
        """
        # Суммируем TDP всех процессоров и видеокарт
        total_tdp = sum(cpu.tdp for cpu in self.get_facts(CPU))
        total_tdp += sum(gpu.tdp for gpu in self.get_facts(GPU))
        base_power = total_tdp + 70          # 70 Вт на остальные компоненты (вентиляторы, SSD, USB)

        # Выбираем коэффициент запаса
        if total_tdp <= 400:
            safety = 1.5
        else:
            safety = 1.75

        # Расчёт в ваттах (сырое значение)
        recommended_raw = int(base_power * safety)
        # Округление до реально существующей мощности БП (минимум 450 Вт)
        recommended = self._round_psu_wattage(recommended_raw)

        # Дополнительное пояснение для мощных видеокарт
        message_suffix = ""
        if any(gpu.tdp > 200 for gpu in self.get_facts(GPU)):
            message_suffix = " (для видеокарт высокого уровня рекомендуется запас 50-75% из‑за пиковых нагрузок)"

        self.clear_facts(PowerStatus)      # удаляем старые статусы мощности
        for psu in self.get_facts(PSU):
            if psu.power >= recommended:
                self.assert_fact(PowerStatus("достаточно", recommended))
                self._replace_psu_message(
                    f"Мощность БП достаточна: {psu.power}W (рекомендовано {recommended}W, расчёт: {recommended_raw}W){message_suffix}"
                )
            else:
                self.assert_fact(PowerStatus("недостаточно", recommended))
                self._replace_psu_message(
                    f"Мощность БП НЕдостаточна: {psu.power}W (рекомендовано {recommended}W, расчёт: {recommended_raw}W){message_suffix}"
                )

    @Rule("R_Final_Build")
    def final_build(self) -> None:
        """
        Итоговое правило: формирует вердикт на основе совместимости по сокету
        и достаточности мощности БП.
        """
        socket_ok = any(f.status == "совместимы" for f in self.get_facts(Compatibility))
        power_ok = any(f.status == "достаточно" for f in self.get_facts(PowerStatus))

        if socket_ok and power_ok:
            self.assert_fact(BuildResult("Сборка корректна"))
        else:
            issues = []
            if not socket_ok:
                issues.append("несовместимые сокеты")
            if not power_ok:
                # Попробуем достать рекомендуемую мощность из любого PowerStatus
                rec = ""
                for p in self.get_facts(PowerStatus):
                    rec = f" (рекомендовано ≥{p.recommended_wattage}W)"
                issues.append(f"недостаточная мощность БП{rec}")
            self.assert_fact(BuildResult(f"Требуется замена компонентов: {', '.join(issues)}"))


# ---------- Функции для ввода и вывода ----------
def read_int(prompt: str, error_message: str) -> int:
    """Безопасно считывает целое число, повторяя запрос при ошибке."""
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print(error_message)


def interactive_components_input() -> tuple[CPU, GPU, Motherboard, PSU]:
    """
    Запрашивает у пользователя характеристики комплектующих для ручной проверки.
    Возвращает кортеж (CPU, GPU, Motherboard, PSU).
    """
    print("\n=== Ввод комплектующих для ручной проверки ===")

    cpu_model = input("Модель CPU: ").strip()
    cpu_socket = input("Сокет CPU: ").strip()
    cpu_tdp = read_int("TDP CPU (Вт): ", "Ошибка: TDP CPU должен быть целым числом.")

    gpu_model = input("Модель GPU: ").strip()
    gpu_tdp = read_int("TDP GPU (Вт): ", "Ошибка: TDP GPU должен быть целым числом.")

    mb_model = input("Модель материнской платы: ").strip()
    mb_socket = input("Сокет материнской платы: ").strip()
    mb_ram_type = input("Тип RAM (DDR4/DDR5/...): ").strip().upper()

    psu_power = read_int("Мощность БП (Вт): ", "Ошибка: мощность БП должна быть целым числом.")

    cpu = CPU(cpu_model, cpu_socket, cpu_tdp)
    gpu = GPU(gpu_model, gpu_tdp)
    motherboard = Motherboard(mb_model, mb_socket, mb_ram_type)
    psu = PSU(psu_power)
    return cpu, gpu, motherboard, psu


def print_selected_components(engine: PCExpertSystem) -> None:
    """Выводит все промежуточные сообщения о выбранных или проверенных компонентах."""
    print("\nПромежуточные факты:")
    for fact in sorted(engine.get_facts(SelectedComponent), key=lambda item: item.description):
        print(f"- {fact.description}")


def print_build_result(engine: PCExpertSystem) -> None:
    """Выводит итоговый результат сборки."""
    print("\nРезультат сборки:")
    for result in engine.get_facts(BuildResult):
        print(f"- {result.status}")


def print_fired_rules(engine: PCExpertSystem) -> None:
    """Выводит список сработавших правил (для отладки и понимания логики)."""
    print("\nСработавшие правила:")
    for rule in engine.fired_rules:
        print(f"- {rule}")


# ---------- Сценарии для демонстрации ----------
def run_scenario(
    name: str,
    motherboard_socket: str,
    motherboard_ram: str,
    psu_power: int,
    purpose: str = "игры",
    budget: int = 80000,
    resolution: str = "1080p",
) -> None:
    """
    Запуск автоматического сценария:
    - система получает требования пользователя (цель, бюджет, разрешение)
    - добавляет материнскую плату и БП из параметров
    - сама выбирает CPU и GPU по правилам R1 и R2
    - выполняет все правила и выводит результаты
    """
    print("\n==============================")
    print(f"Сценарий: {name}")
    print("==============================")

    engine = PCExpertSystem()
    engine.assert_fact(UserRequirement(purpose, budget, resolution))
    engine.assert_fact(Motherboard("MSI B550", motherboard_socket, motherboard_ram))
    engine.assert_fact(PSU(psu_power))
    engine.run()

    print_selected_components(engine)
    print_build_result(engine)
    print_fired_rules(engine)


def run_manual_configuration(
    name: str,
    cpu: CPU,
    gpu: GPU,
    motherboard: Motherboard,
    psu: PSU,
) -> None:
    """
    Ручной режим: пользователь сам задаёт все компоненты.
    Система только проверяет совместимость и достаточность БП.
    """
    print("\n==============================")
    print(f"Сценарий: {name}")
    print("==============================")

    engine = PCExpertSystem()
    engine.assert_fact(cpu)
    engine.assert_fact(gpu)
    engine.assert_fact(motherboard)
    engine.assert_fact(psu)
    engine.run()

    print_selected_components(engine)
    print_build_result(engine)
    print_fired_rules(engine)


def main() -> None:
    """Точка входа: выбор режима работы (ручной или автоматические сценарии)."""
    choice = input("Вы хотите ввести комплектующие вручную? (y/n): ").strip().lower()
    if choice == "y":
        cpu, gpu, motherboard, psu = interactive_components_input()
        run_manual_configuration("Ручная конфигурация", cpu, gpu, motherboard, psu)
        return

    # Демонстрационные автоматические сценарии (исправлены и дополнены)
    # 1. Корректная бюджетная сборка, где хватает 450 Вт
    run_scenario("Бюджетная сборка (450W достаточно)", "AM4", "DDR4", 450, budget=40000, resolution="1080p")
    # 2. Корректная средняя сборка, требуется 650 Вт
    run_scenario("Средняя сборка (650W)", "AM4", "DDR4", 650, budget=70000, resolution="1080p")
    # 3. Недостаточный БП 450W для мощной видеокарты (должен выдать ошибку)
    run_scenario("Мощная сборка с БП 450W (недостаточно)", "AM4", "DDR4", 450, budget=90000, resolution="1440p")
    # 4. Несовместимый сокет (LGA1200 вместо AM4)
    run_scenario("Несовместимый сокет LGA1200", "LGA1200", "DDR4", 650)
    # 5. Очень слабая сборка (без дискретной видеокарты?) – но у нас всегда есть GPU по правилам, поэтому просто показываем минимальный порог


if __name__ == "__main__":
    main()