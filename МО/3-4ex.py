class PCEngine:
    def __init__(self):
        self.facts = {}
        self.rules_triggered = []
        self.recommendations = {}

        self.min_budgets = {
            'офис': 25000,
            'игры': 75000,
            'pro': 300000
        }

        self.db = {
            'CPU': [
                {'name': 'Intel Core i9-14900K', 'socket': 'LGA1700', 'type': 'игры', 'tdp': 125, 'price': 62000},
                {'name': 'Intel Core i7-13700K', 'socket': 'LGA1700', 'type': 'игры', 'tdp': 125, 'price': 45000},
                {'name': 'Intel Core i5-13400F', 'socket': 'LGA1700', 'type': 'игры', 'tdp': 65, 'price': 22000},
                {'name': 'Intel Core i5-12400F', 'socket': 'LGA1700', 'type': 'игры', 'tdp': 65, 'price': 14000},
                {'name': 'Intel Core i3-12100F', 'socket': 'LGA1700', 'type': 'игры', 'tdp': 60, 'price': 8000},
                {'name': 'Intel Core i3-12100', 'socket': 'LGA1700', 'type': 'офис', 'tdp': 60, 'price': 11000},
                {'name': 'Intel Pentium Gold G7400', 'socket': 'LGA1700', 'type': 'офис', 'tdp': 46, 'price': 7000},
                {'name': 'AMD Ryzen 9 7950X3D', 'socket': 'AM5', 'type': 'игры', 'tdp': 120, 'price': 68000},
                {'name': 'AMD Ryzen 7 7800X3D', 'socket': 'AM5', 'type': 'игры', 'tdp': 120, 'price': 42000},
                {'name': 'AMD Ryzen 5 7600X', 'socket': 'AM5', 'type': 'игры', 'tdp': 105, 'price': 24000},
                {'name': 'AMD Ryzen 5 8500G', 'socket': 'AM5', 'type': 'офис', 'tdp': 65, 'price': 18000},
                {'name': 'AMD Ryzen 7 5800X', 'socket': 'AM4', 'type': 'игры', 'tdp': 105, 'price': 21000},
                {'name': 'AMD Ryzen 5 5600', 'socket': 'AM4', 'type': 'игры', 'tdp': 65, 'price': 13500},
                {'name': 'AMD Ryzen 5 5500', 'socket': 'AM4', 'type': 'игры', 'tdp': 65, 'price': 11000},
                {'name': 'AMD Ryzen 5 4500', 'socket': 'AM4', 'type': 'офис', 'tdp': 65, 'price': 8000},
                {'name': 'AMD Athlon 3000G', 'socket': 'AM4', 'type': 'офис', 'tdp': 35, 'price': 5500},
                {'name': 'AMD Ryzen Threadripper 3970X', 'socket': 'sTRX4', 'type': 'pro', 'tdp': 280, 'price': 150000},
                {'name': 'AMD Ryzen Threadripper 3960X', 'socket': 'sTRX4', 'type': 'pro', 'tdp': 280, 'price': 90000},
                {'name': 'Intel Xeon W-3275M', 'socket': 'LGA3647', 'type': 'pro', 'tdp': 205, 'price': 120000},
                {'name': 'Intel Core i9-10980XE', 'socket': 'LGA2066', 'type': 'pro', 'tdp': 165, 'price': 70000}
            ],
            'MB': [
                {'name': 'ASUS ROG STRIX Z790-E', 'socket': 'LGA1700', 'ram_type': 'DDR5', 'size': 'ATX',
                 'price': 48000},
                {'name': 'MSI MAG B760 TOMAHAWK', 'socket': 'LGA1700', 'ram_type': 'DDR5', 'size': 'ATX',
                 'price': 21000},
                {'name': 'Gigabyte B760M DS3H', 'socket': 'LGA1700', 'ram_type': 'DDR4', 'size': 'mATX',
                 'price': 12500},
                {'name': 'ASRock H610M-HVS', 'socket': 'LGA1700', 'ram_type': 'DDR4', 'size': 'mATX', 'price': 7500},
                {'name': 'GIGABYTE X670E AORUS', 'socket': 'AM5', 'ram_type': 'DDR5', 'size': 'ATX', 'price': 35000},
                {'name': 'ASUS TUF GAMING B650-PLUS', 'socket': 'AM5', 'ram_type': 'DDR5', 'size': 'ATX',
                 'price': 23000},
                {'name': 'MSI PRO A620M-E', 'socket': 'AM5', 'ram_type': 'DDR5', 'size': 'mATX', 'price': 10000},
                {'name': 'MSI MPG X570S', 'socket': 'AM4', 'ram_type': 'DDR4', 'size': 'ATX', 'price': 19000},
                {'name': 'ASUS PRIME B550M-K', 'socket': 'AM4', 'ram_type': 'DDR4', 'size': 'mATX', 'price': 10500},
                {'name': 'Gigabyte A520M K V2', 'socket': 'AM4', 'ram_type': 'DDR4', 'size': 'mATX', 'price': 6500},
                {'name': 'ASUS ROG Zenith II Extreme', 'socket': 'sTRX4', 'ram_type': 'DDR4', 'size': 'ATX',
                 'price': 75000},
                {'name': 'Supermicro X11DPG-QT', 'socket': 'LGA3647', 'ram_type': 'DDR4', 'size': 'ATX',
                 'price': 90000},
                {'name': 'ASUS WS X299 SAGE', 'socket': 'LGA2066', 'ram_type': 'DDR4', 'size': 'ATX', 'price': 45000}
            ],
            'RAM': [
                {'name': 'Corsair Vengeance 64GB (2x32) 6000MHz', 'type': 'DDR5', 'price': 38000},
                {'name': 'Kingston FURY Beast 32GB (2x16) 5600MHz', 'type': 'DDR5', 'price': 22000},
                {'name': 'Crucial 32GB (2x16) 4800MHz', 'type': 'DDR5', 'price': 15000},
                {'name': 'G.Skill Aegis 32GB (2x16) 3200MHz', 'type': 'DDR4', 'price': 11000},
                {'name': 'ADATA XPG 16GB (2x8) 3200MHz', 'type': 'DDR4', 'price': 6000},
                {'name': 'Patriot Signature 16GB (2x8) 2666MHz', 'type': 'DDR4', 'price': 5000}
            ],
            'GPU': [
                {'name': 'NVIDIA RTX 4090 24GB', 'tdp': 450, 'type': 'игры', 'price': 210000},
                {'name': 'NVIDIA RTX 4080 Super 16GB', 'tdp': 320, 'type': 'игры', 'price': 125000},
                {'name': 'AMD Radeon RX 7900 XTX 24GB', 'tdp': 355, 'type': 'игры', 'price': 110000},
                {'name': 'NVIDIA RTX 5070 12GB', 'tdp': 220, 'type': 'игры', 'price': 80000},
                {'name': 'NVIDIA RTX 4070 12GB', 'tdp': 200, 'type': 'игры', 'price': 65000},
                {'name': 'NVIDIA RTX 5060 Ti 16GB', 'tdp': 180, 'type': 'игры', 'price': 45000},
                {'name': 'NVIDIA RTX 5060 8GB', 'tdp': 150, 'type': 'игры', 'price': 33000},
                {'name': 'NVIDIA RTX 5050 8GB', 'tdp': 130, 'type': 'игры', 'price': 27000},
                {'name': 'AMD Radeon RX 7600 8GB', 'tdp': 165, 'type': 'игры', 'price': 32000},
                {'name': 'NVIDIA GTX 1650 4GB', 'tdp': 75, 'type': 'игры', 'price': 15000},
                {'name': 'Integrated Graphics', 'tdp': 0, 'type': 'офис', 'price': 0},
                {'name': 'NVIDIA RTX A6000 48GB', 'tdp': 300, 'type': 'pro', 'price': 300000},
                {'name': 'NVIDIA RTX 5000 Ada 32GB', 'tdp': 250, 'type': 'pro', 'price': 220000},
                {'name': 'AMD Radeon Pro W6800 32GB', 'tdp': 250, 'type': 'pro', 'price': 150000},
                {'name': 'NVIDIA RTX 4000 SFF 20GB', 'tdp': 130, 'type': 'pro', 'price': 80000},
                {'name': 'NVIDIA Quadro P5000 16GB', 'tdp': 180, 'type': 'pro', 'price': 50000}
            ],
            'Storage': [
                {'name': 'Samsung 980 Pro 1TB NVMe', 'type': 'NVMe', 'interface': 'PCIe 4.0', 'price': 12000},
                {'name': 'WD Black SN850X 1TB NVMe', 'type': 'NVMe', 'interface': 'PCIe 4.0', 'price': 13000},
                {'name': 'Crucial P3 Plus 1TB NVMe', 'type': 'NVMe', 'interface': 'PCIe 4.0', 'price': 8000},
                {'name': 'Kingston NV3 1TB NVMe', 'type': 'NVMe', 'interface': 'PCIe 4.0', 'price': 7500},
                {'name': 'Samsung 870 EVO 1TB SATA SSD', 'type': 'SSD', 'interface': 'SATA', 'price': 9000},
                {'name': 'Kingston A400 480GB SATA SSD', 'type': 'SSD', 'interface': 'SATA', 'price': 4000},
                {'name': 'Seagate BarraCuda 2TB HDD', 'type': 'HDD', 'interface': 'SATA', 'price': 6000},
                {'name': 'Western Digital Blue 1TB HDD', 'type': 'HDD', 'interface': 'SATA', 'price': 5000}
            ],
            'Case': [
                {'name': 'Be Quiet! Silent Base 802', 'size': 'ATX', 'price': 19000},
                {'name': 'Lian Li LANCOOL II', 'size': 'ATX', 'price': 12000},
                {'name': 'Deepcool MATREXX 55', 'size': 'ATX', 'price': 6500},
                {'name': 'AeroCool Cylon Mini', 'size': 'mATX', 'price': 3500}
            ],
            'PSU': [
                {'name': 'Super Flower Leadex 1000W', 'power': 1000, 'price': 22000},
                {'name': 'Corsair RM850x 850W', 'power': 850, 'price': 16000},
                {'name': 'Be Quiet! Pure Power 700W', 'power': 700, 'price': 11000},
                {'name': 'Deepcool PK550D 550W', 'power': 550, 'price': 5000},
                {'name': 'Aerocool VX 400W', 'power': 400, 'price': 2500}
            ]
        }

        self.rules = []
        rule_names = [
            'rule_cpu',
            'rule_gpu',
            'rule_motherboard',
            'rule_ram',
            'rule_storage',
            'rule_case',
            'rule_psu']
        for name in rule_names:
            attr = getattr(self, name, None)
            if attr and callable(attr) and hasattr(attr, '_is_rule'):
                self.rules.append(attr)

    def add_fact(self, key, value):
        self.facts[key] = value

    # Декоратор для регистрации правил (устанавливает флаг)
    @staticmethod
    def rule(func):
        func._is_rule = True
        return func

    # Правило R1: подбор CPU
    @rule
    def rule_cpu(self):
        purpose = self.facts['purpose']
        budget = self.facts['budget']

        if purpose == 'игры':
            cpu_share = 0.15
        elif purpose == 'офис':
            cpu_share = 0.30
        else:  # pro
            cpu_share = 0.25

        cpu_limit = budget * cpu_share
        available_cpus = sorted([c for c in self.db['CPU']
                                 if c['type'] == purpose and c['price'] <= cpu_limit],
                                key=lambda x: x['price'], reverse=True)
        if not available_cpus:
            return f"Не удалось подобрать CPU под лимит {cpu_limit} руб."
        self.recommendations['CPU'] = available_cpus[0]
        self.rules_triggered.append(
            f"R1: Выбран CPU {
            self.recommendations['CPU']['name']}")
        return True

    # Правило R5: подбор GPU
    @rule
    def rule_gpu(self):
        purpose = self.facts['purpose']
        budget = self.facts['budget']

        if purpose == 'игры':
            gpu_share = 0.45
        elif purpose == 'офис':
            self.recommendations['GPU'] = next(
                g for g in self.db['GPU'] if g['name'] == 'Integrated Graphics')
            self.rules_triggered.append("R5: Выбрана встроенная графика")
            return True
        else:
            gpu_share = 0.30

        gpu_limit = budget * gpu_share
        available_gpus = sorted([g for g in self.db['GPU']
                                 if g['type'] == purpose and g['price'] <= gpu_limit],
                                key=lambda x: x['price'], reverse=True)
        if not available_gpus:
            return f"Не удалось подобрать GPU под лимит {gpu_limit} руб."
        self.recommendations['GPU'] = available_gpus[0]
        self.rules_triggered.append(
            f"R5: Выбрана видеокарта {
            self.recommendations['GPU']['name']}")
        return True

    # Правило R3: подбор материнской платы (совместимость по сокету)
    @rule
    def rule_motherboard(self):
        budget = self.facts['budget']
        purpose = self.facts['purpose']

        if purpose == 'игры':
            mb_share = 0.10
        elif purpose == 'офис':
            mb_share = 0.15
        else:  # pro
            mb_share = 0.10

        mb_limit = budget * mb_share
        cpu_socket = self.recommendations['CPU']['socket']
        available_mbs = sorted([m for m in self.db['MB']
                                if m['socket'] == cpu_socket and m['price'] <= mb_limit],
                               key=lambda x: x['price'], reverse=True)
        if not available_mbs:
            available_mbs = sorted(
                [m for m in self.db['MB'] if m['socket'] == cpu_socket], key=lambda x: x['price'])
            if not available_mbs:
                return "Совместимых плат не найдено."
        self.recommendations['MB'] = available_mbs[0]
        self.rules_triggered.append(
            f"R3: Выбрана MB {
            self.recommendations['MB']['name']}")
        return True

    # Правило R8: подбор ОЗУ (совместимость по типу памяти)
    @rule
    def rule_ram(self):
        budget = self.facts['budget']
        purpose = self.facts['purpose']

        if purpose == 'игры':
            ram_share = 0.10
        elif purpose == 'офис':
            ram_share = 0.12
        else:  # pro
            ram_share = 0.12

        ram_limit = budget * ram_share
        ram_type = self.recommendations['MB']['ram_type']
        available_ram = sorted([r for r in self.db['RAM']
                                if r['type'] == ram_type and r['price'] <= ram_limit],
                               key=lambda x: x['price'], reverse=True)
        if not available_ram:
            available_ram = sorted(
                [r for r in self.db['RAM'] if r['type'] == ram_type], key=lambda x: x['price'])
        self.recommendations['RAM'] = available_ram[0]
        self.rules_triggered.append(
            f"R8: Выбрана ОЗУ {
            self.recommendations['RAM']['name']}")
        return True

    # Правило R12: подбор накопителя
    @rule
    def rule_storage(self):
        budget = self.facts['budget']
        purpose = self.facts['purpose']

        # [citation:1]
        if purpose == 'игры':
            storage_share = 0.10  # уменьшено с 0.15
        elif purpose == 'офис':
            storage_share = 0.20
        else:  # pro
            storage_share = 0.17

        storage_limit = budget * storage_share
        if purpose == 'игры' or purpose == 'pro':
            preferred_types = ['NVMe', 'SSD']
        else:
            preferred_types = ['SSD', 'HDD']

        available_storage = []
        for t in preferred_types:
            candidates = [s for s in self.db['Storage']
                          if s['type'] == t and s['price'] <= storage_limit]
            if candidates:
                available_storage = sorted(
                    candidates, key=lambda x: x['price'], reverse=True)
                break
        if not available_storage:
            available_storage = sorted(
                self.db['Storage'], key=lambda x: x['price'])
        self.recommendations['Storage'] = available_storage[0]
        self.rules_triggered.append(
            f"R12: Выбран накопитель {
            self.recommendations['Storage']['name']}")
        return True

    # Правило R11: подбор корпуса (совместимость по форм-фактору)
    @rule
    def rule_case(self):
        budget = self.facts['budget']
        purpose = self.facts['purpose']

        if purpose == 'игры':
            case_share = 0.05
        elif purpose == 'офис':
            case_share = 0.10
        else:
            case_share = 0.03

        case_limit = budget * case_share
        mb_size = self.recommendations['MB']['size']
        available_cases = sorted(
            [
                c for c in self.db['Case'] if (
                                                      c['size'] == 'ATX' or mb_size == 'mATX') and c[
                                                  'price'] <= case_limit],
            key=lambda x: x['price'],
            reverse=True)
        if not available_cases:
            available_cases = sorted(self.db['Case'], key=lambda x: x['price'])
        self.recommendations['Case'] = available_cases[0]
        self.rules_triggered.append(
            f"R11: Выбран корпус {
            self.recommendations['Case']['name']}")
        return True

    # Правило R6: подбор блока питания
    @rule
    def rule_psu(self):
        budget = self.facts['budget']
        purpose = self.facts['purpose']

        if purpose == 'игры':
            psu_share = 0.05
        elif purpose == 'офис':
            psu_share = 0.10
        else:  # pro
            psu_share = 0.03

        total_tdp = self.recommendations['CPU']['tdp'] + \
                    self.recommendations['GPU']['tdp'] + 50
        required_power = total_tdp * 1.25
        psu_limit = budget * psu_share

        available_psus = sorted([p for p in self.db['PSU']
                                 if p['power'] >= required_power and p['price'] <= psu_limit],
                                key=lambda x: x['price'], reverse=True)
        if not available_psus:
            available_psus = sorted([p for p in self.db['PSU'] if p['power'] >= required_power],
                                    key=lambda x: x['price'])
        self.recommendations['PSU'] = available_psus[0]
        self.rules_triggered.append(
            f"R6: Выбран БП {
            self.recommendations['PSU']['name']}")
        return True

    def run(self):
        """Запуск механизма вывода: последовательное выполнение правил"""
        if 'purpose' not in self.facts or 'budget' not in self.facts:
            return "Отсутствуют исходные факты (purpose, budget)"

        for rule in self.rules:
            result = rule()
            if isinstance(result, str):
                return result
        return True

    def show(self):
        print("\n" + "=" * 30)
        print("ИТОГОВАЯ СБОРКА:")
        total_price = 0
        for part, item in self.recommendations.items():
            print(f"- {part}: {item['name']} ({item['price']} руб.)")
            total_price += item['price']
        print(f"ОБЩАЯ СТОИМОСТЬ: {total_price} руб.")
        print("=" * 30)
        print("Цепочка правил:",
              " -> ".join([r.split(':')[0] for r in self.rules_triggered]))


def get_valid_input(prompt, options=None, is_numeric=False):
    while True:
        user_input = input(prompt).strip().lower()
        if is_numeric:
            if user_input.isdigit() and int(user_input) > 0:
                return int(user_input)
            print("Ошибка: Введите положительное число.")
        elif options:
            if user_input in options:
                return user_input
            print(f"Ошибка: Выберите из списка {options}")
        else:
            return user_input


def get_budget_with_min(purpose, min_budget):
    while True:
        budget = get_valid_input(
            f"Введите бюджет на основные компоненты (руб) - минимум для {purpose} {min_budget} руб: ",
            is_numeric=True)
        if budget >= min_budget:
            return budget
        else:
            print(
                f"Ошибка: Для цели '{purpose}' минимальный бюджет составляет {min_budget} руб. Пожалуйста, введите больше.")


def main():
    engine = PCEngine()
    print("Система экспертного подбора комплектующих")

    purpose = get_valid_input(
        "Введите цель (игры/офис/pro): ",
        options=[
            'игры',
            'офис',
            'pro'])
    min_budget = engine.min_budgets[purpose]
    budget = get_budget_with_min(purpose, min_budget)

    engine.add_fact('purpose', purpose)
    engine.add_fact('budget', budget)

    result = engine.run()
    if result is True:
        engine.show()
    else:
        print(f"\nРезультат: {result}")


if __name__ == "__main__":
    main()
