-- train.lua
-- Trains the intent classifier and saves weights + vocab to disk.
-- Run once (or whenever you add new training data).

package.path = package.path .. ";./?.lua"

local nn      = require "nn"
local dataset = require "dataset"

print("╔══════════════════════════════════════╗")
print("║  VOSS — Training Run  v2             ║")
print("╚══════════════════════════════════════╝")
print()

-- ── Build vocabulary ──────────────────────────────────────────────────────────

io.write("Building vocabulary (unigrams + bigrams)... ")
local vocab, word2idx = dataset.build_vocab()
local vocab_size      = #vocab
print(string.format("done. %d tokens.", vocab_size))

local vf = assert(io.open("vocab.dat", "w"))
for _, w in ipairs(vocab) do vf:write(w .. "\n") end
vf:close()

-- ── Build training set ────────────────────────────────────────────────────────

io.write("Encoding training samples... ")
local train_data = dataset.build(word2idx, vocab_size)
local n_intents  = #dataset.intents
print(string.format("done. %d samples, %d classes.", #train_data, n_intents))

local inf = assert(io.open("intents.dat", "w"))
for _, intent in ipairs(dataset.intents) do inf:write(intent .. "\n") end
inf:close()

-- ── Build network ─────────────────────────────────────────────────────────────

local h1, h2, h3 = 128, 64, 32

print(string.format("\nArchitecture: %d → %d → %d → %d → %d",
    vocab_size, h1, h2, h3, n_intents))

local net = nn.new(
    { vocab_size, h1, h2, h3, n_intents },
    { "lrelu", "lrelu", "lrelu", "sigmoid" }
)

-- ── Progress display ──────────────────────────────────────────────────────────

local function fmt_time(s)
    s = math.floor(s)
    if s < 60 then
        return string.format("%ds", s)
    elseif s < 3600 then
        return string.format("%dm%02ds", math.floor(s / 60), s % 60)
    else
        return string.format("%dh%02dm", math.floor(s / 3600), math.floor((s % 3600) / 60))
    end
end

local BAR_WIDTH = 28

local function draw_bar(epoch, total, loss, lr, phase_start)
    local frac    = epoch / total
    local filled  = math.floor(frac * BAR_WIDTH)
    local bar     = string.rep("\xe2\x96\x88", filled) .. string.rep("\xe2\x96\x91", BAR_WIDTH - filled)
    local elapsed = os.clock() - phase_start
    local eta     = frac > 0 and (elapsed / frac * (1 - frac)) or 0
    io.write(string.format(
        "\r  [%s] %4d/%-4d  loss: %.5f  lr: %.5f  elapsed: %s  eta: %s   ",
        bar, epoch, total, loss, lr,
        fmt_time(elapsed), fmt_time(eta)
    ))
    io.flush()
end

local function draw_empty_bar(total, lr)
    local bar = string.rep("\xe2\x96\x91", BAR_WIDTH)
    io.write(string.format(
        "  [%s]    0/%-4d  loss: -.-----  lr: %.5f  elapsed: 0s  eta: ?   ",
        bar, total, lr
    ))
    io.flush()
end

-- ── Training ──────────────────────────────────────────────────────────────────

local function run_phase(label, epochs, lr, opts)
    print("\n" .. label)
    local cur_lr       = lr
    local momentum     = opts.momentum     or 0.9
    local dropout_rate = opts.dropout_rate or 0.0
    local decay        = opts.decay        or 1.0
    local min_lr       = opts.min_lr       or 1e-5
    local patience     = opts.patience     or 0
    local phase_start  = os.clock()

    draw_empty_bar(epochs, cur_lr)

    local best_loss  = math.huge
    local no_improve = 0

    for epoch = 1, epochs do
        local total_loss = 0.0

        for i = #train_data, 2, -1 do
            local j = math.random(i)
            train_data[i], train_data[j] = train_data[j], train_data[i]
        end

        for _, sample in ipairs(train_data) do
            local cache = net:forward(sample.input, dropout_rate)
            local out   = cache[net.n_layers].a
            local loss  = 0.0
            for k = 1, #out do
                loss = loss + (out[k] - sample.target[k]) ^ 2
            end
            total_loss = total_loss + loss / #out
            net:backward(cache, sample.target, cur_lr, momentum)
        end

        if decay < 1.0 then
            cur_lr = math.max(min_lr, cur_lr * decay)
        end

        local avg_loss = total_loss / #train_data
        draw_bar(epoch, epochs, avg_loss, cur_lr, phase_start)

        if patience > 0 then
            if avg_loss < best_loss - 1e-6 then
                best_loss  = avg_loss
                no_improve = 0
            else
                no_improve = no_improve + 1
                if no_improve >= patience then
                    print()
                    print(string.format("  Early stop at epoch %d (no improvement for %d epochs).",
                        epoch, patience))
                    break
                end
            end
        end
    end

    print()
    print(string.format("  Done in %s.", fmt_time(os.clock() - phase_start)))
end

-- ── Run phases ────────────────────────────────────────────────────────────────

local t0 = os.clock()

run_phase(
    "Phase 1 -- warm-up  (300 epochs, lr=0.01, momentum=0.85, no dropout)",
    300, 0.01,
    { momentum = 0.85, dropout_rate = 0.0 }
)

run_phase(
    "Phase 2 -- fine-tune  (1200 epochs, lr=0.005, momentum=0.9, dropout=0.2, decay)",
    1200, 0.005,
    { momentum = 0.9, dropout_rate = 0.2, decay = 0.9995, min_lr = 1e-5, patience = 80 }
)

print(string.format("\nTotal training time: %s", fmt_time(os.clock() - t0)))

-- ── Validation ────────────────────────────────────────────────────────────────

io.write("\nValidation on training set: ")
local correct, total = 0, #train_data
local per_intent_correct = {}
local per_intent_total   = {}
for _, intent in ipairs(dataset.intents) do
    per_intent_correct[intent] = 0
    per_intent_total[intent]   = 0
end

for _, sample in ipairs(train_data) do
    local out     = net:predict(sample.input)
    local pred, _ = net:argmax(out)
    local targ, _ = net:argmax(sample.target)
    local intent  = dataset.intents[targ]
    per_intent_total[intent]   = (per_intent_total[intent]   or 0) + 1
    if pred == targ then
        correct = correct + 1
        per_intent_correct[intent] = (per_intent_correct[intent] or 0) + 1
    end
end

print(string.format("%.1f%% (%d/%d)", correct / total * 100, correct, total))

print("\nPer-intent accuracy:")
for _, intent in ipairs(dataset.intents) do
    local c   = per_intent_correct[intent] or 0
    local t   = per_intent_total[intent]   or 0
    local pct = t > 0 and (c / t * 100) or 0
    local bar = string.rep("\xe2\x96\x88", math.floor(pct / 5)) .. string.rep("\xe2\x96\x91", 20 - math.floor(pct / 5))
    print(string.format("  %-14s %s %3.0f%%  (%d/%d)", intent, bar, pct, c, t))
end

-- ── Save ──────────────────────────────────────────────────────────────────────

net:save("voss_weights.dat")
print("\nWeights  --> voss_weights.dat")
print("Vocab    --> vocab.dat")
print("Intents  --> intents.dat")
print("\nRun  luajit main.lua  to start chatting.")