import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import pytextrank
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Tribal.utils.text_cluster import TextCluster
from detoxify import Detoxify
from nltk.tokenize import word_tokenize
from TRUNAJOD import surface_proxies, ttr
from nltk.util import ngrams
from collections import Counter
from nltk import pos_tag
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from easyLLM.easyLLM import EasyLLM
import gc

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Dataset class for loading data
class ExtremistDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, word2vec_model=None, use_word2vec=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.use_word2vec = use_word2vec

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.use_word2vec:
            tokens = text.lower().split()
            embeddings = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
            if not embeddings:
                embeddings = [np.zeros(self.word2vec_model.vector_size)]
            embeddings = torch.tensor(embeddings, dtype=torch.float)
            return embeddings, label
        else:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}
            inputs['labels'] = torch.tensor(label, dtype=torch.long)
            return inputs

# Custom collate function for DataLoader
def collate_fn(batch):
    embeddings = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([len(x) for x in embeddings], dtype=torch.long)
    embeddings_padded = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    return embeddings_padded, labels, lengths

# FeatureExtractor class
class FeatureExtractor:
    VECTOR_SIZE = 100
    _idf_model = None
    _word2vec_model = None

    def __init__(self, list_of_text_for_topic_creation, list_of_baseline_posts_for_vec_model):
        self._build_idf_model(list_of_text_for_topic_creation)
        self._build_word_2_vec_model(list_of_baseline_posts_for_vec_model)
        self.llm = EasyLLM()

        # Lazy load spacy and models
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

        # Initialize VADER and Detoxify models
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.detoxify_model = Detoxify('original')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize hate speech lexicon (Replace with your provided list)
        self.hate_speech_terms = set(["Alert","Aim","Automatic","Anguish","Agitator","Apartheid","Agency","Aircraft","Airplane","Acid","Airport","Aerial","Assassinate","Account","Arms","Assault","Ambush","Anarchy","Authority","Aggressor","Allies","Alarm","Ashore","Atrocity","Artillery","Airfield","Annihilate","Appeasement","Arsenal","Attrition","Aggression","Armory","Ammunition","Advance","Assassin","Armedforces","Alliance","Attack","Armament","Bloodletting","Bulletproof","Brutal","Betray","Betrayal","Blood(y)","Boobytrap","Bombardment","Battalion","Bullet","Brute","Burn","Brutality","Bully","Blowup","Bunker","Booby trap","Blast","Bomb","Breach","Belligerent","Battle","Bury","Bloody","Blood","Blindside","Burning","Barrage","Barricade","Battlefield","Break","Conspiracy","Clash","Conspire","Coordinate","Civilian","Cautionary","Chief","Coalition","Camouflage","Captive","Coordinates","Corps","Carrier","Control","Concentration","Carnage","Conquer","Clamor","Compassion","Compliance","Crash","Checkpoint","Clandestine","Chopper","Confrontation","Causes","Countermand","Conflict","Crime","Counterattack","Courageous","Chaos","Commandos","Casualties","Confrontation(al)","Cautious","Consequences","Consolidate","Convoy","Checking","Crisis","Confusion","Cataclysm","Careen","Command(or)","Combat","Charred","Collapse","Cross-hairs","Capture","Culpability","Corpse","Cargo","Cadaver","Charge","Concussion","Campaign","Conflagration","Deliberate","Devastation","Discipline","Disperse","Dispatch","Dead","Death","Defensive","Dominate","Drone","Detect","Danger","Detection","Deploy","Detonate","Destruction","Demolish","Demoralize","Damage","Defend","Deception","Drama","Disaster","Dictator","Despot","Disease","Device","Domination","Duck","Duty","Debris","Dash","Decline","Defiant","Dictatorship","Defect","Doom","Disastrous","Division","Die","Downfall","Dispute","Desert","Disruption","Disarray","Dissonance","Dread","Defense","Dismantle","Dangerous","Deadly","Destroy","Demoralization","Debacle","Disarmament","Enemy","Expunge","Evacuate","Escalate","Explosion","Execute","Excess","Extremism","Evacuee","Explosive","Execution","Epithet","Exploitation","Enforce","Exercise","Explode","Expectations","Encounter","Engagement","Escape","Escalation","Enforcement","Endurance","Force(s)","Faction","Force","Fierce","Flight","Fortification","Flank","Ferment","Frenzy","Feud","Front lines","Fray","Fear","Fearless","Felon","Fugitive","Fright","Forceful","Furtive","Fuel","Fighter","Fanatic","Fiery","Fearful","Forces","Flee","Fatal","Frontlines","Foxhole","Ferocious","Fight","Gas","Germ warfare","Grenade","Guided bombs","Grave","Gang up on","Garrison","Guard","Generator","Germwarfare","Groans","Gunship","Government","Gang","Genocide","Grievous","Guerrillas","Guidedbombs","Guns","Hazard","Harass","Heroic","Hide","Hostility","Horses","Horror","Horrific","Harsh","Hit","Hiding","Helicopter","Heroism","Hijack","Hostile","Hijacker","Hatred","Hit-and-run","Howitzer","Hurt","Hatch","Holocaust","Hammering","Hate","Involvement","International","Interdiction","Infanticide","Ire","Invasion","Incident","Interrogation","Ignite","Instructions","Intimidate","Insurrection","Inflame","Inferred","Intense","Incontrovertible","Impact","Informant","Investigate","Intelligence","Improvise","Incite","Intercept","Infantry","Investigations","Infiltrate","Injuries","Inmate","Intervene","Insurgent","Jail","Join","Jets","Jeer","Knock-out","Keening","Knife","Kamikaze","Kidnap","Knives","Keen","Kill","Killing","Lamentation","Legacy","Liaison","Loathsome","Loyalty","Landmines","Laser-activated","Liberation","Linksto","Launcher","Liberators","Launch","Method","Militaristic","Mobile","Militant","Massacre","Menace","Malicious","Military","Momentum","Mines","Militancy","Maim","Militia","Mob","Mobilization","Machines","Mortars","Machineguns","March","Megalomania","Mission","Mayhem","Muscle","Murder","Missile","Mistreatment","Malevolent","Munitions","Maraud","Notorious","Nationalist","Negotiation","Nightmare","Nitrate","Neutralize","Overthrow","Onerous","Out of control","Operation","Officials","Offensive","Order","Overrun","Opposition","Outbreak","Planes","Prisoner","Pilot","Prowl","Post-traumatic","Pugnacious","Partisan","Premeditate","Prey","Patriotism","Plunder","Paramedics","Platoon","Potent","Powder","Power","Pacify","Persecute","Penetration","Pound","Provocation","Pistol","Performance","Patriot","Proliferation","Penetrate","Pushing","Pulverize","Preemptive","Petrify","Prison","Perform","Position","Photos","Patrol","Powerful","Quarrel","Quail","Quiver","Quell","Rally","Refugee","Revenge","Radical","Reputation","Retreat","Ravish","Revolution","Retribution","Radiation","Relentless","Rift","Rule","Resistance","Rounds","Recovery","Rebellion","Reparation","Retaliation","Reaction","Readiness","Recruitment","Reconnaissance","Regiment","Rot","Recruit","Reinforcements","Reprisal","Rival","Ricochet","Ravage","Rocket","Ruthless","Rescue","Rage","Rebel","Rifle","Riot","Regime","Shot","Strategy","Smash","Survival","Survivor","Showdown","Supplies","Sacrifice","Stronghold","Surrender","Storage","Salvage","Sanction","Strength","Surprise","Security","Seize","Secrecy","Seizure","Strife","Siege","Sensor","Secret","Stash","Scramble","Storm","Shock","Shells","Sedition","Skirmish","Strip","Suppression","Strangle","Special-ops","Shoot","Smuggle","Slaughter","Score","Sabotage","Spokesman","Soldier","Savage","Superstition","Suffering","Squad","Strategist","Specialized","Stalk","Struggle","Straggler","Subversive","Support","Stealth","Spysatellite","Strategic","Shelling","Spy","Screening","Strike","Setback","Spotter","Scare","Spy satellite","Submarine","Tsunami","Tactics","Triumph","Training","Tragic","Trauma","Torch","Terrorism","Threat","Terrorize","Thug","Torpedo","Tension","Turbulent","Tornado","Trigger","Trench","Tank","Terror","Topple","Tourniquet","Target","Terrain","Thwart","Treachery","Transportation","Trample","Trap","Terrorist","Threaten","Uprising","Urgency","Unruly","Unite","Unleash","Unify","Unit","Unexpected","Unbelievable","Uniform","Unconventional","Vociferous","Virulence","Violence","Vulnerability","Vow","Venomous","Victory","Vanguard","Vehicular","Vital","Vicious","Violation","Vanish","Veteran","Vehicle","Void","Vile","Vitriol","Vagrant","Vilify","Vendetta","Watchful","Warnings","Weather","Watchlist","Wince","Warplane","Watchdog","Weapon","Well-trained","Worldwide","Wreckage","Wage","Wound","Warrior","Wounds","Whiz","Warrant","Warheads","War","Wisdom","X-ray","Yearn","Yelling","Zigzag","Zeal","Zealot","Zone","pedophile","child molester","demonic","scumbag","fucking","demon-god","daemon"])

        # Build TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(list_of_baseline_posts_for_vec_model)

    ######################################
    # Feature extraction methods remain unchanged...
    
    def get_capital_letter_word_frequency(self, text):
        tokens = word_tokenize(text)
        total_words = len(tokens)
        capital_words = [word for word in tokens if word.isupper()]
        capital_word_count = len(capital_words)
        if total_words == 0:
            return 0
        return capital_word_count / total_words

    def get_pos_counts(self, text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        return dict(pos_counts)

    def get_n_grams(self, text, n=2, top_n=10):
        tokens = word_tokenize(text.lower())
        n_grams = ngrams(tokens, n)
        n_gram_counts = Counter(n_grams)
        most_common_n_grams = n_gram_counts.most_common(top_n)
        return [' '.join(gram) for gram, count in most_common_n_grams]

    def get_emotion_scores(self, text):
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores

    def get_hate_speech_lexicon_counts(self, text):
        tokens = word_tokenize(text.lower())
        hate_terms_in_text = [token for token in tokens if token in self.hate_speech_terms]
        hate_term_counts = Counter(hate_terms_in_text)
        return dict(hate_term_counts)

    def get_tf_idf_vector(self, text):
        tfidf_vector = self.tfidf_vectorizer.transform([text])
        return tfidf_vector

    def get_text_topic(self, text):
        dominant_topic = self._idf_model.add_new_text(text)
        return dominant_topic

    def get_words_per_sentence(self, text):
        doc = self.nlp(text)
        total_sentences = len(list(doc.sents))
        total_words = len([token for token in doc if token.is_alpha])
        if total_sentences == 0:
            return 0
        return total_words / total_sentences

    def get_syntactical_complexity(self, text):
        doc = self.nlp(text)
        clause_count = surface_proxies.clause_count(doc)
        return clause_count

    def get_lexical_diversity(self, text):
        doc = self.nlp(text)
        lexical_diversity = ttr.lexical_diversity_mtld(doc)
        return lexical_diversity

    def get_readability(self, text):
        doc = self.nlp(text)
        readability_index = surface_proxies.readability_index(doc)
        return readability_index

    def get_toxicity(self, text):
        return self.detoxify_model.predict(text)

    def get_sentiment(self, text):
        return self.sentiment_analyzer.polarity_scores(text)

    def get_embeddings(self, text):
        embeddings = self.embedding_model.encode([text])
        return embeddings

    def get_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def get_keywords(self, text, top_n=10):
        doc = self.nlp(text)
        phrases = [(phrase.text, phrase.rank) for phrase in doc._.phrases]
        sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        top_phrases = sorted_phrases[:top_n]
        top_words = [phrase[0] for phrase in top_phrases]
        return top_words

    def get_text_vector(self, text):
        tokens = word_tokenize(text.lower())
        vector = np.zeros((self.VECTOR_SIZE,))
        count = 0

        for token in tokens:
            if token in self._word2vec_model.wv:
                vector += self._word2vec_model.wv[token]
                count += 1

        if count > 0:
            vector /= count

        return vector

    @staticmethod
    def is_text_valid(text):
        tokens = word_tokenize(text)
        return len(tokens) > 5

    ################################

    def _build_idf_model(self, list_of_all_text, num_topics=5):
        self._idf_model = TextCluster(list_of_all_text, num_topics)

    def _build_word_2_vec_model(self, list_of_baseline_posts_for_vec_model):
        tokenized_texts = [word_tokenize(text.lower()) for text in list_of_baseline_posts_for_vec_model]
        model = Word2Vec(min_count=1, window=5, vector_size=self.VECTOR_SIZE)
        model.build_vocab(tokenized_texts)
        model.train(tokenized_texts, total_examples=model.corpus_count, epochs=model.epochs)
        self._word2vec_model = model