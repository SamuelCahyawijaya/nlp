# coding=utf-8

import pandas as pd
import pyarrow as pa

import nlp

_CITATION = """
# TODO
"""

_DESCRIPTION = """
IndoNLU, the Language Understanding Evaluation benchmark for Indonesian language
(https://indobenchmark.com/) is a collection of resources for training,
evaluating, and analyzing natural language understanding systems for Indonesian language.
"""

class IndoNLUConfig(nlp.BuilderConfig):
    """BuilderConfig for IndoNLU."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for GLUE.

    Args:
      text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
      label_column: `string`, name of the column in the tsv file corresponding
        to the label
      data_url: `string`, url to download the zip file from
      data_dir: `string`, the path to the folder containing the tsv files in the
        downloaded zip
      citation: `string`, citation for the data set
      url: `string`, url for information about the data set
      label_classes: `list[string]`, the list of classes if the label is
        categorical. If not provided, then the label will be of type
        `nlp.Value('float32')`.
      process_label: `Function[string, any]`, function  taking in the raw value
        of the label and processing it to the form required by the label feature
      **kwargs: keyword arguments forwarded to super.
    """
        super(GlueConfig, self).__init__(
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"), **kwargs
        )
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label
                
class IndoNLU(nlp.GeneratorBasedBuilder):
    """IndoNLU Benchmerk Dataset."""
    
    VERSION = nlp.Version("1.0.0")
    BUILDER_CONFIGS = [
        IndoNLUConfig(
            name="smsa",
            description=textwrap.dedent(
                """\
            Sentence-level sentiment analysis corpus consisting collection of comments and reviews in Indonesian 
            obtained from multiple online platforms, such as Twitter, Zomato, TripAdvisor, Facebook, Instagram, and Qraved. 
            The text was crawled and then annotated by several Indonesian linguists to construct this dataset.
            There are three possible sentiments on the SmSA dataset: positive, negative, and neutral."""
            ),
            text_features=['sentence'],
            label_classes=["unacceptable", "acceptable"],
            label_column="label",
            data_url="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",
            data_dir="CoLA",
            citation=textwrap.dedent(
                """\
            @article{warstadt2018neural,
              title={Neural Network Acceptability Judgments},
              author={Warstadt, Alex and Singh, Amanpreet and Bowman, Samuel R},
              journal={arXiv preprint arXiv:1805.12471},
              year={2018}
            }"""
            ),
            url="https://nyu-mll.github.io/CoLA/",
        ),
        IndoNLUConfig(
            name="nerp",
            description=textwrap.dedent(
                """\
            Named entity recognition corpus contains texts collected from several Indonesian news websites. 
            The labels consists of 5 name entitiies: 
            PER (person), LOC (location), IND (product or brand), EVT (event), and FNB (food and beverage).
            The label of the dataset is spans in IOB chunking representation"""
            ),
            text_features=['sentence'],
            label_classes=['I-PPL','B-EVT','B-PLC','I-IND','B-IND','B-FNB','I-EVT','B-PPL','I-PLC','O','I-FNB'],
            label_column="label",
            data_url="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",
            data_dir="SST-2",
            citation=textwrap.dedent(
                """\
            @inproceedings{socher2013recursive,
              title={Recursive deep models for semantic compositionality over a sentiment treebank},
              author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},
              booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
              pages={1631--1642},
              year={2013}
            }"""
            ),
            url="https://nlp.stanford.edu/sentiment/index.html",
        )
    ]
    
    def _info(self):
        features = {text_feature: nlp.Value("string") for text_feature in six.iterkeys(self.config.text_features)}
        if self.config.label_classes:
            features["label"] = nlp.features.ClassLabel(names=self.config.label_classes)
        else:
            features["label"] = nlp.Value("float32")
        features["idx"] = nlp.Value("int32")
        return nlp.DatasetInfo(
            description=_GLUE_DESCRIPTION,
            features=nlp.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _GLUE_CITATION,
        )

    def _split_generators(self, dl_manager):
        """ We handle string, list and dicts in datafiles
        """
        if isinstance(self.config.data_files, (str, list, tuple)):
            files = self.config.data_files
            if isinstance(files, str):
                files = [files]
            return [nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name in [nlp.Split.TRAIN, nlp.Split.VALIDATION, nlp.Split.TEST]:
            if split_name in self.config.data_files:
                files = self.config.data_files[split_name]
                if isinstance(files, str):
                    files = [files]
                splits.append(nlp.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_tables(self, files):
        for i, file in enumerate(files):
            pa_table = pa.Table.from_pandas(pd.read_pickle(file))
            yield i, pa_table