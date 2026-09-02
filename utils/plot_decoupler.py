from mad_gui.plugins import BaseAlgorithm


class DecouplePlotsAlgorithm(BaseAlgorithm):
    @classmethod
    def name(cls) -> str:
        return "Desacoplar Zoom de Gráficas (Ejes Independientes)"

    @classmethod
    def description(cls) -> str:
        return "Desvincula el zoom y paneo horizontal entre los sensores SENS y Matrix."

    def process_data(self, data: dict) -> dict:
        # En mad-gui, self.parent o el entorno gráfico contiene la referencia a MainWindow
        parent = getattr(self, "parent", None)
        main_window = None

        if parent and hasattr(parent, "plots"):
            main_window = parent
        elif hasattr(self, "main_window"):
            main_window = getattr(self, "main_window")

        if main_window and hasattr(main_window, "plots"):
            # Usamos .values() para cumplir con la regla PERF102 de Ruff
            for plot_widget in main_window.plots.values():
                if hasattr(plot_widget, "plot_item"):
                    vb = plot_widget.plot_item.getViewBox()
                    # Rompemos el enlace horizontal con la primera gráfica
                    vb.setXLink(None)
                    vb.setYLink(None)

        return data