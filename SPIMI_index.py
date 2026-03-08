#-------------------------------------------------------------
# AUTHOR: Seanglong Lim
# FILENAME: SPIMI_index.py
# SPECIFICATION: Build a SPIMI algorithm to extract data from corpus.tsv
# FOR: CS 5180- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10

READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


# ---------------------------------------------------------
# 1) READ FIRST BLOCK OF 100 DOCUMENTS USING PANDAS
# ---------------------------------------------------------
# Use pandas.read_csv with chunksize=100.
# Each chunk corresponds to one memory block.
# Convert docIDs like "D0001" to integers.
# ---------------------------------------------------------

chunk_iter = pd.read_csv(
    INPUT_PATH,
    sep="\t",
    header=None,
    names=["doc_id", "text"],
    chunksize=BLOCK_SIZE,
    encoding="utf-8"
)
# ---------------------------------------------------------
# 2) BUILD PARTIAL INDEX (SPIMI STYLE) FOR CURRENT BLOCK
# ---------------------------------------------------------
# - Use CountVectorizer(stop_words='english')
# - Fit and transform the 100 documents
# - Reconstruct binary postings lists from the sparse matrix
# - Store postings in a dictionary: term -> set(docIDs)
# ---------------------------------------------------------

def build_partial_index(doc_ids, documents):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    partial_index = {}
    coo = X.tocoo()
    for row, col, value in zip(coo.row, coo.col, coo.data):
        if value > 0:
            term = terms[col]
            docid = doc_ids[row]
            if term not in partial_index:
                partial_index[term] = set()
            partial_index[term].add(docid)
    return partial_index
# ---------------------------------------------------------
# 3) FLUSH PARTIAL INDEX TO DISK
# ---------------------------------------------------------
# - Sort terms lexicographically
# - Sort postings lists (ascending docID)
# - Write to: block_1.txt, block_2.txt, ..., block_10.txt
# - Format: term:docID1,docID2,docID3
# ---------------------------------------------------------

def write_block_file(block_number, partial_index):
    filename = f"block_{block_number}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted(partial_index.keys()):
            postings = sorted(partial_index[term])
            postings_str = ",".join(str(docid) for docid in postings)
            f.write(f"{term}:{postings_str}\n")


# ---------------------------------------------------------
# 4) REPEAT STEPS 1–3 FOR ALL 10 BLOCKS
# ---------------------------------------------------------
# - Continue reading next 100-doc chunks
# - After processing each block, flush to disk
# - Do NOT keep previous blocks in memory
# ---------------------------------------------------------

block_count = 0

for chunk in chunk_iter:
    block_count += 1
    if block_count > NUM_BLOCKS:
        break
    doc_ids = [int(str(x).strip().replace("D", "")) for x in chunk["doc_id"]]
    documents = chunk["text"].fillna("").astype(str).tolist()
    partial_index = build_partial_index(doc_ids, documents)
    write_block_file(block_count, partial_index)
if block_count != NUM_BLOCKS:
    print(f"Warning: expected {NUM_BLOCKS} blocks, but processed {block_count} block(s).")

# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------
# After all block files are created:
# - Open block_1.txt ... block_10.txt simultaneously
# ---------------------------------------------------------
block_files = []

for i in range(1, NUM_BLOCKS + 1):
    f = open(f"block_{i}.txt", "r", encoding="utf-8")
    block_files.append(f)


# ---------------------------------------------------------
# 6) INITIALIZE READ BUFFERS
# ---------------------------------------------------------
# For each block file:
# - Read up to READ_BUFFER_LINES_PER_FILE lines
# - Parse each line into (term, postings_list)
# - Store in a per-file read buffer
# ---------------------------------------------------------

read_buffers = []

buffer_positions = []

for f in block_files:
    buffer_list = []
    for _ in range(READ_BUFFER_LINES_PER_FILE):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line:
            term, postings_str = line.split(":")
            postings = [int(x) for x in postings_str.split(",")] if postings_str else []
            buffer_list.append((term, postings))

    read_buffers.append(buffer_list)
    buffer_positions.append(0)


# ---------------------------------------------------------
# 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
# ---------------------------------------------------------
# - Push the first term from each read buffer into a min-heap
# - Heap elements: (term, file_index)
# ---------------------------------------------------------

heap = []

for file_index in range(NUM_BLOCKS):
    if len(read_buffers[file_index]) > 0:
        first_term = read_buffers[file_index][0][0]
        heapq.heappush(heap, (first_term, file_index))


# ---------------------------------------------------------
# 8) MERGE LOOP
# ---------------------------------------------------------
# While min-heap is not empty:
#   1. Pop the min-heap root (smallest term)
#   2. Keep popping the min-heap root while the current term equals the previous term
#   3. Collect all read buffers whose current term matches
#   4. Merge postings lists associated with this term (sorted + deduplicated)
#   5. Advance corresponding read buffer pointers
#   6. If a read buffer is exhausted, read next 100 lines from the corresponding block (if available)
#   7. For each read buffer whose pointer advanced, push its new pointed term into the heap (if available).
# ---------------------------------------------------------

write_buffer = []

with open("final_index.txt", "w", encoding="utf-8") as final_out:
    while heap:
        current_term, file_index = heapq.heappop(heap)
        same_term_entries = [file_index]
        merged_postings = set()
        while heap and heap[0][0] == current_term:
            _, other_file_index = heapq.heappop(heap)
            same_term_entries.append(other_file_index)
        for fi in same_term_entries:
            pos = buffer_positions[fi]
            term, postings = read_buffers[fi][pos]
            merged_postings.update(postings)
            buffer_positions[fi] += 1
            if buffer_positions[fi] >= len(read_buffers[fi]):
                new_buffer = []
                for _ in range(READ_BUFFER_LINES_PER_FILE):
                    line = block_files[fi].readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        term2, postings_str2 = line.split(":")
                        postings2 = [int(x) for x in postings_str2.split(",")] if postings_str2 else []
                        new_buffer.append((term2, postings2))
                read_buffers[fi] = new_buffer
                buffer_positions[fi] = 0

            if buffer_positions[fi] < len(read_buffers[fi]):
                next_term = read_buffers[fi][buffer_positions[fi]][0]
                heapq.heappush(heap, (next_term, fi))


# ---------------------------------------------------------
# 9) WRITE BUFFER MANAGEMENT
# ---------------------------------------------------------
# - Append merged term-line to write buffer
# - If write buffer reaches WRITE_BUFFER_LINES:
#       flush (append) to final_index.txt
# - After merge loop ends:
#       flush remaining write buffer
# ---------------------------------------------------------
        merged_postings = sorted(merged_postings)
        postings_str = ",".join(str(docid) for docid in merged_postings)
        write_buffer.append(f"{current_term}:{postings_str}")

        if len(write_buffer) >= WRITE_BUFFER_LINES:
            final_out.write("\n".join(write_buffer) + "\n")
            write_buffer = []

    if write_buffer:
        final_out.write("\n".join(write_buffer) + "\n")



# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------
# - Close all open block files
# - Ensure final_index.txt is properly written
# ---------------------------------------------------------
for f in block_files:
    f.close()

print("Done.")
print("Created block_1.txt ... block_10.txt and final_index.txt")