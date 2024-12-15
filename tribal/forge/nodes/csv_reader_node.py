from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseSourceNode
import csv
import time

class CsvReaderNode(BaseSourceNode):
    def __init__(self, broadcast_manager, csv_path):

        self.csv_path = csv_path

        super().__init__("CSV Reader", broadcast_manager)

    def _run(self):

       with open(self.csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:

                time.sleep(0.1)
                message = self._construct_message(self.name, dict(row))
                self.send_broadcast(message, self.name)