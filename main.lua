-- main.lua
-- VOSS — interactive chat loop.
-- Loads trained weights, vocab, and persona; runs a REPL.

package.path = package.path .. ";./?.lua"

local nn      = require "nn"
local dataset = require "dataset"
local persona = require "persona"

-- ─── Load vocab ───────────────────────────────────────────────────────────────

local function load_lines(path)
    local t = {}
    local f = assert(io.open(path, "r"), "Missing file: " .. path
        .. "\n  → Run: lua5.4 train.lua first.")
    for line in f:lines() do t[#t + 1] = line end
    f:close()
    return t
end

local vocab_list = load_lines("vocab.dat")
local intent_list = load_lines("intents.dat")

local word2idx = {}
for i, w in ipairs(vocab_list) do word2idx[w] = i end

local intent_by_idx = {}
for i, v in ipairs(intent_list) do intent_by_idx[i] = v end

-- ─── Load network ─────────────────────────────────────────────────────────────

local net = nn.load("voss_weights.dat")

-- ─── Load persistent memory ───────────────────────────────────────────────────

persona.memory_load()

-- ─── Helpers ──────────────────────────────────────────────────────────────────

local function classify(text)
    local vec = dataset.encode(text, word2idx, #vocab_list)
    local out = net:predict(vec)
    local idx, conf = net:argmax(out)
    local intent = intent_by_idx[idx] or "unknown"

    -- Build sorted confidence table for display
    local scores = {}
    for i, v in ipairs(out) do
        scores[#scores + 1] = { intent = intent_by_idx[i], score = v }
    end
    table.sort(scores, function(a, b) return a.score > b.score end)

    return intent, conf, scores
end

local function bar(v, width)
    width = width or 20
    local filled = math.floor(v * width + 0.5)
    return ("█"):rep(filled) .. ("░"):rep(width - filled)
end

local function mood_bar(mood)
    local markers = { "😤", "😑", "😐", "🙂", "😊" }
    local idx = math.max(1, math.min(5, math.floor(mood * 5) + 1))
    return markers[idx] .. string.format(" %.0f%%", mood * 100)
end

-- ─── Header ───────────────────────────────────────────────────────────────────

local function header()
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  VOSS  ·  Artificial. Not artificial-nice.           ║")
    print("║  Pure-Lua neural net · intent classifier · memory    ║")
    print("╚══════════════════════════════════════════════════════╝")
    local state = persona.get_state()
    if state.user_name then
        print("  Remembered from last session: " .. state.user_name)
    end
    print("  Type 'quit' or 'exit' to leave.")
    print("  Type '/debug' to toggle confidence scores.")
    print("  Type '/memory' to see what VOSS remembers.")
    print("  Type '/fact <text>' to store a fact about yourself.")
    print("  Type '/forget <keyword>' to remove a stored fact.")
    print("  Type '/stats' to see session stats.")
    print()
end

-- ─── REPL ─────────────────────────────────────────────────────────────────────

header()

local debug_mode = false
local turn       = 0

while true do
    io.write("you  › ")
    io.flush()
    local line = io.read("l")
    if not line then break end
    line = line:match("^%s*(.-)%s*$")  -- trim
    if line == "" then goto continue end

    -- Commands
    if line == "quit" or line == "exit" then
        print()
        local resp, _, _ = persona.respond("farewell", line, 1.0)
        print("VOSS › " .. resp)
        print()
        break

    elseif line == "/debug" then
        debug_mode = not debug_mode
        print("  [debug " .. (debug_mode and "ON" or "OFF") .. "]")
        goto continue

    elseif line == "/memory" then
        local st = persona.get_state()
        local function pct(v) return string.format("%.0f%%", v * 100) end
        print("  ┌─ VOSS memory ─────────────────────")
        print("  │ Name       : " .. (st.user_name or "(unknown)"))
        print("  │ Mood       : " .. mood_bar(st.mood))
        print("  │ Trust      : " .. pct(st.trust))
        print("  │ Turn       : " .. st.turn)
        print("  │ All-time   : " .. (st.total_turns or 0))
        local facts = persona.recall_facts()
        if #facts == 0 then
            print("  │ Facts      : (none stored)")
        else
            print("  │ Facts      :")
            for _, f in ipairs(facts) do
                print("  │   · " .. f)
            end
        end
        print("  └────────────────────────────────────")
        goto continue

    elseif line:match("^/fact%s+(.+)$") then
        local fact = line:match("^/fact%s+(.+)$")
        persona.remember(fact)
        print("  [Stored: '" .. fact .. "']")
        goto continue

    elseif line:match("^/forget%s+(.+)$") then
        local kw = line:match("^/forget%s+(.+)$")
        local removed = persona.forget(kw)
        if removed > 0 then
            print("  [Removed " .. removed .. " fact(s) matching '" .. kw .. "']")
        else
            print("  [No facts matched '" .. kw .. "']")
        end
        goto continue

    elseif line == "/stats" then
        local s = persona.get_stats()
        local function pct(v) return string.format("%.0f%%", v * 100) end
        print("  ┌─ VOSS session stats ───────────────")
        print("  │ Name         : " .. (s.name or "(unknown)"))
        print("  │ Session turns: " .. s.session_turns)
        print("  │ All-time turns: " .. s.total_turns)
        print("  │ Session time : " .. s.elapsed)
        print("  │ Mood         : " .. pct(s.mood))
        print("  │ Trust        : " .. pct(s.trust) .. " (" .. s.trust_label .. ")")
        print("  │ Stored facts : " .. s.facts_count)
        print("  └────────────────────────────────────")
        goto continue
    end

    -- Classify
    turn = turn + 1
    local intent, confidence, scores = classify(line)
    local resp, mood, _ = persona.respond(intent, line, confidence)

    -- Response
    print()
    print("VOSS › " .. resp)
    print()

    -- Debug panel
    if debug_mode then
        print("  ┌─ debug ───────────────────────────────────────")
        print(string.format("  │ Intent : %-12s  conf: %.3f", intent, confidence))
        print(string.format("  │ Mood   : %s", mood_bar(mood)))
        print("  │ Top scores:")
        for i = 1, math.min(5, #scores) do
            local s = scores[i]
            print(string.format("  │   %-12s %s %.3f",
                s.intent, bar(s.score, 14), s.score))
        end
        print("  └────────────────────────────────────────────────")
        print()
    end

    ::continue::
end

print("Session ended. VOSS is going quiet.")