Real-Word IR Text Collection (DocID, Text)

Format requested: DocID Text
- corpus.tsv is tab-separated: DocID<TAB>Text (one document per line)

Files:
- corpus.tsv        : DocID<TAB>Text
- metadata.jsonl    : doc_id, topics, length_tokens
- vocab.txt         : all terms observed in corpus (sorted by descending frequency)

Statistics:
- Documents: 1000
- Distinct terms observed: 8795
- Total tokens: 174156
- Average doc length (tokens): 174.2

Vocabulary source:
- Base vocabulary drawn from the "google-10000-english" 10,000 common English words list.

Notes:
- Documents are synthetic but composed entirely of real English words from the word list.
- Each document is lightly "topic-biased" by injecting topic-specific words, then filling the remainder with Zipf-like sampling for realistic term frequency skew.
