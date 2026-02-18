import torch
import torch.nn as nn

# ============================================================================
# LSTM STEP-BY-STEP EXPLANATION
# ============================================================================

print("\n" + "="*70)
print("3️⃣ HOW LSTM WORKS: Step-by-Step Example")
print("="*70)

# ============================================================================
# Simple Scenario: Generate caption "dog running"
# ============================================================================

print("""
📝 SCENARIO: Generate caption word-by-word

Caption to generate: "<start> dog running <end>"

The LSTM processes this SEQUENTIALLY:
  Time step 1: Input "<start>" → Predict "dog"
  Time step 2: Input "dog"     → Predict "running"
  Time step 3: Input "running" → Predict "<end>"
""")

# ============================================================================
# Setup
# ============================================================================

vocab = {"<start>": 0, "dog": 1, "running": 2, "<end>": 3}
vocab_size = 4
embed_size = 3
hidden_size = 4

embedding = nn.Embedding(vocab_size, embed_size)
lstm_cell = nn.LSTMCell(embed_size, hidden_size)  # Single LSTM cell for clarity

print("\n🔧 LSTM Configuration:")
print(f"  Input size (embedding): {embed_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Vocabulary size: {vocab_size}")

# ============================================================================
# Initialize LSTM State
# ============================================================================

print("\n" + "="*70)
print("🎬 INITIALIZATION")
print("="*70)

# LSTM has TWO states:
# 1. Hidden state (h) - short-term memory (what to output NOW)
# 2. Cell state (c) - long-term memory (what to remember for LATER)

h_t = torch.zeros(1, hidden_size)  # Hidden state
c_t = torch.zeros(1, hidden_size)  # Cell state

print(f"\nInitial hidden state (h_0): {h_t.shape}")
print(f"  {h_t}")
print(f"\nInitial cell state (c_0): {c_t.shape}")
print(f"  {c_t}")

print("\n💡 Think of it as:")
print("  h_t = 'Current thought' (short-term)")
print("  c_t = 'Long-term memory' (context)")

# ============================================================================
# TIME STEP 1: Process "<start>"
# ============================================================================

print("\n" + "="*70)
print("⏰ TIME STEP 1: Input word '<start>'")
print("="*70)

# Get embedding for "<start>"
word_idx = torch.tensor([0])  # <start> = 0
x_1 = embedding(word_idx)  # (1, embed_size)

print(f"\n📥 Input:")
print(f"  Word: '<start>' (index={word_idx.item()})")
print(f"  Embedding: {x_1.shape}")
print(f"  {x_1}")

print(f"\n🧠 LSTM Processing:")
print(f"  Input:  x_1 = {x_1.shape}  (word embedding)")
print(f"  State:  h_0 = {h_t.shape}  (previous hidden)")
print(f"  Memory: c_0 = {c_t.shape}  (previous cell)")

# LSTM processes the input
h_1, c_1 = lstm_cell(x_1, (h_t, c_t))

print(f"\n📤 Output:")
print(f"  New hidden state (h_1): {h_1.shape}")
print(f"    {h_1}")
print(f"  New cell state (c_1): {c_1.shape}")
print(f"    {c_1}")

print(f"\n💭 What happened?")
print(f"  - LSTM looked at '<start>' embedding")
print(f"  - Updated its memory (cell state)")
print(f"  - Generated new hidden state")
print(f"  - h_1 will be used to PREDICT next word: 'dog'")

# ============================================================================
# TIME STEP 2: Process "dog"
# ============================================================================

print("\n" + "="*70)
print("⏰ TIME STEP 2: Input word 'dog'")
print("="*70)

word_idx = torch.tensor([1])  # dog = 1
x_2 = embedding(word_idx)

print(f"\n📥 Input:")
print(f"  Word: 'dog' (index={word_idx.item()})")
print(f"  Embedding: {x_2.shape}")

print(f"\n🧠 LSTM Processing:")
print(f"  Input:  x_2 = {x_2.shape}  (current word)")
print(f"  State:  h_1 = {h_1.shape}  (from previous step)")
print(f"  Memory: c_1 = {c_1.shape}  (remembers '<start>')")

# LSTM processes the input
h_2, c_2 = lstm_cell(x_2, (h_1, c_1))

print(f"\n📤 Output:")
print(f"  New hidden state (h_2): {h_2.shape}")
print(f"    {h_2}")
print(f"  New cell state (c_2): {c_2.shape}")
print(f"    {c_2}")

print(f"\n💭 What happened?")
print(f"  - LSTM looked at 'dog' embedding")
print(f"  - Remembered context from previous step (that we started)")
print(f"  - Updated memory to include 'dog'")
print(f"  - h_2 will predict next word: 'running'")

# ============================================================================
# TIME STEP 3: Process "running"
# ============================================================================

print("\n" + "="*70)
print("⏰ TIME STEP 3: Input word 'running'")
print("="*70)

word_idx = torch.tensor([2])  # running = 2
x_3 = embedding(word_idx)

print(f"\n📥 Input:")
print(f"  Word: 'running' (index={word_idx.item()})")

print(f"\n🧠 LSTM Processing:")
print(f"  Input:  x_3 = {x_3.shape}  (current word)")
print(f"  State:  h_2 = {h_2.shape}  (remembers '<start> dog')")
print(f"  Memory: c_2 = {c_2.shape}  (full context)")

# LSTM processes the input
h_3, c_3 = lstm_cell(x_3, (h_2, c_2))

print(f"\n📤 Output:")
print(f"  New hidden state (h_3): {h_3.shape}")
print(f"  New cell state (c_3): {c_3.shape}")

print(f"\n💭 What happened?")
print(f"  - LSTM looked at 'running'")
print(f"  - Has full context: '<start> dog running'")
print(f"  - h_3 will predict next word: '<end>'")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("📊 LSTM SEQUENTIAL PROCESSING VISUALIZATION")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────┐
│                  LSTM Processing Timeline                    │
└─────────────────────────────────────────────────────────────┘

Time Step 1:
  Input: "<start>"
  ┌─────────────┐
  │ LSTM Cell   │
  │ h_0 = zeros │ ──→ Process "<start>" ──→ h_1 (remembers start)
  │ c_0 = zeros │ ──→ Update memory    ──→ c_1 (context: "beginning")
  └─────────────┘
  Output: h_1 → Predicts "dog"

Time Step 2:
  Input: "dog" + h_1 (previous state) + c_1 (previous memory)
  ┌─────────────┐
  │ LSTM Cell   │
  │ h_1         │ ──→ Process "dog" ──→ h_2 (remembers start + dog)
  │ c_1         │ ──→ Update memory ──→ c_2 (context: "dog is subject")
  └─────────────┘
  Output: h_2 → Predicts "running"

Time Step 3:
  Input: "running" + h_2 + c_2
  ┌─────────────┐
  │ LSTM Cell   │
  │ h_2         │ ──→ Process "running" ──→ h_3 (full context)
  │ c_2         │ ──→ Update memory     ──→ c_3 (context: "action done")
  └─────────────┘
  Output: h_3 → Predicts "<end>"

🔑 Key Point: Each step DEPENDS on the previous step!
            The LSTM "remembers" what it has seen before.
""")

# ============================================================================
# THE MAGIC: Memory Gates
# ============================================================================

print("="*70)
print("🎩 THE MAGIC: How LSTM Remembers (Gates)")
print("="*70)

print("""
LSTM has 3 gates that control information flow:

1️⃣ FORGET GATE (f_t):
   "What should I forget from old memory?"
   
   Example at step 2:
   - Saw "<start>" before
   - Now seeing "dog"
   - Forget gate: "Keep '<start>' info (f=0.8), it's still relevant"
   
   Formula: f_t = sigmoid(W_f · [h_{t-1}, x_t])
   Output: 0.0 (forget everything) to 1.0 (remember everything)

2️⃣ INPUT GATE (i_t):
   "What new information should I remember?"
   
   Example at step 2:
   - Current word: "dog"
   - Input gate: "This is important! Remember 'dog' (i=0.9)"
   
   Formula: i_t = sigmoid(W_i · [h_{t-1}, x_t])
   Output: 0.0 (ignore new info) to 1.0 (remember strongly)

3️⃣ OUTPUT GATE (o_t):
   "What should I output right now?"
   
   Example at step 2:
   - Context: "start + dog"
   - Output gate: "Output info about what comes AFTER 'dog' (o=0.85)"
   - Result: Predicts "running"
   
   Formula: o_t = sigmoid(W_o · [h_{t-1}, x_t])
   Output: 0.0 (output nothing) to 1.0 (output everything)

📝 CELL STATE UPDATE:
   c_t = (f_t * c_{t-1}) + (i_t * candidate_memory)
        ↑                 ↑
   Keep old memory   Add new memory
   
📝 HIDDEN STATE UPDATE:
   h_t = o_t * tanh(c_t)
         ↑           ↑
    Output gate   Current memory
""")

# ============================================================================
# CONCRETE EXAMPLE WITH NUMBERS
# ============================================================================

print("="*70)
print("🔢 CONCRETE EXAMPLE WITH NUMBERS")
print("="*70)

print("""
Scenario: LSTM at time step 2, processing "dog"

Previous state:
  h_1 = [0.1, 0.3, -0.2, 0.5]  (remembers "<start>")
  c_1 = [0.2, 0.4, 0.1, 0.3]   (memory: "sentence beginning")

Current input:
  x_2 = [0.8, 0.6, 0.4]        ("dog" embedding)

Gates compute:
  Forget gate:  f_2 = [0.8, 0.9, 0.7, 0.85]  ← Keep most of old memory
  Input gate:   i_2 = [0.9, 0.8, 0.7, 0.9]   ← New "dog" info important
  Candidate:    c̃_2 = [0.6, 0.7, 0.5, 0.6]   ← New memory candidate
  Output gate:  o_2 = [0.85, 0.8, 0.75, 0.9] ← What to output

Cell state update:
  c_2 = (f_2 * c_1) + (i_2 * c̃_2)
      = ([0.8, 0.9, 0.7, 0.85] * [0.2, 0.4, 0.1, 0.3])
        + ([0.9, 0.8, 0.7, 0.9] * [0.6, 0.7, 0.5, 0.6])
      = [0.16, 0.36, 0.07, 0.26] + [0.54, 0.56, 0.35, 0.54]
      = [0.70, 0.92, 0.42, 0.80]  ← Updated memory (old + new)

Hidden state update:
  h_2 = o_2 * tanh(c_2)
      = [0.85, 0.8, 0.75, 0.9] * tanh([0.70, 0.92, 0.42, 0.80])
      = [0.85, 0.8, 0.75, 0.9] * [0.60, 0.73, 0.40, 0.66]
      = [0.51, 0.58, 0.30, 0.59]  ← Output for prediction

This h_2 goes to Linear layer → Predicts "running"!
""")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("="*70)
print("💡 KEY INSIGHTS")
print("="*70)

print("""
1. ✅ LSTM processes words SEQUENTIALLY (one at a time)
2. ✅ Maintains TWO states:
   - Cell state (c): Long-term memory (the "story so far")
   - Hidden state (h): Short-term output (what to say next)
3. ✅ Uses GATES to control information flow:
   - Forget: What to forget from old memory
   - Input: What new info to remember
   - Output: What to output right now
4. ✅ Each step builds on the previous step
5. ✅ This allows it to remember CONTEXT across many words!

🎯 Why LSTM for caption generation?
   - Remembers image context (from encoder)
   - Remembers previous words (via cell state)
   - Generates coherent, contextual sentences
   - Can handle long captions (20+ words)

🔄 The Flow:
   Encoder → Context vector → LSTM initial state
   LSTM step 1 → Generate word 1
   LSTM step 2 → Generate word 2 (remembers word 1)
   LSTM step 3 → Generate word 3 (remembers words 1 & 2)
   ...
   Until <end> token
""")