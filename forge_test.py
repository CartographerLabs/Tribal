from tribal.forge.nodes import *
from tribal.forge.managers import BroadcastManager

broadcast_manager = BroadcastManager()

csv_node = CsvReaderNode(broadcast_manager, "test_posts.csv")
csv_node.set_message_type("post")

json_reader_node = JsonReaderNode(broadcast_manager, r"C:\Users\JS\Downloads\mini-tribe\content\output.json")
json_reader_node.set_message_type("post")

feature_extractor_node = FeatureExtractorNode(broadcast_manager)
feature_extractor_node.add_broadcast_implicit_receiver("post")
feature_extractor_node.set_message_type("feature_extracted_posts")

json_out_node = JsonOutNode(broadcast_manager, "out.json")
json_out_node.add_explicit_receiver(feature_extractor_node.name)


decision_node = DecisionNode(broadcast_manager, expression="message['toxicity'] > 0.9")
decision_node.add_broadcast_implicit_receiver("feature_extracted_posts")
decision_node.set_message_type("high_sentiment")

alert_out_node = AlertOutNode(broadcast_manager)
alert_out_node.add_broadcast_implicit_receiver("high_sentiment")

json_reader_node.start()
#csv_node.start()