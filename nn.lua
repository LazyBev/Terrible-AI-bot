-- nn.lua
-- Pure Lua neural network: feedforward, backprop, weight persistence
-- Improvements: momentum SGD, gradient clipping, LR decay, better init

local nn = {}

-- ─── Math helpers ────────────────────────────────────────────────────────────

local function sigmoid(x)
    return 1.0 / (1.0 + math.exp(-math.max(-500, math.min(500, x))))
end

local function sigmoid_d(x)
    return x * (1.0 - x)
end

local function tanh_act(x)
    return math.tanh(x)
end

local function tanh_d(x)
    return 1.0 - x * x
end

local function relu(x)
    return x > 0 and x or 0.0
end

local function relu_d(x)
    return x > 0 and 1.0 or 0.0
end

-- Leaky ReLU: avoids dead neurons
local function lrelu(x)
    return x > 0 and x or 0.01 * x
end

local function lrelu_d(x)
    return x > 0 and 1.0 or 0.01
end

local function softmax(vec)
    local max = vec[1]
    for i = 2, #vec do if vec[i] > max then max = vec[i] end end
    local s, out = 0.0, {}
    for i = 1, #vec do
        out[i] = math.exp(vec[i] - max)
        s = s + out[i]
    end
    for i = 1, #vec do out[i] = out[i] / s end
    return out
end

local ACTS = {
    sigmoid  = { f = sigmoid,  d = sigmoid_d  },
    tanh     = { f = tanh_act, d = tanh_d     },
    relu     = { f = relu,     d = relu_d     },
    lrelu    = { f = lrelu,    d = lrelu_d    },
    linear   = { f = function(x) return x end, d = function(_) return 1.0 end },
}

-- ─── Network constructor ──────────────────────────────────────────────────────

function nn.new(layer_sizes, activations)
    local net = {
        layers   = {},
        n_layers = #layer_sizes - 1,
    }

    math.randomseed(os.time())

    for i = 1, net.n_layers do
        local fan_in  = layer_sizes[i]
        local fan_out = layer_sizes[i + 1]
        local act     = activations[i] or "sigmoid"

        -- He init for relu/lrelu, Xavier for others
        local scale
        if act == "relu" or act == "lrelu" then
            scale = math.sqrt(2.0 / fan_in)
        else
            scale = math.sqrt(2.0 / (fan_in + fan_out))  -- Xavier
        end

        local W, b = {}, {}
        -- Momentum accumulators
        local vW, vb = {}, {}

        for o = 1, fan_out do
            W[o]  = {}
            vW[o] = {}
            for inp = 1, fan_in do
                -- Box-Muller for Gaussian init
                local u1 = math.max(1e-10, math.random())
                local u2 = math.random()
                local gauss = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                W[o][inp]  = gauss * scale
                vW[o][inp] = 0.0
            end
            b[o]  = 0.0
            vb[o] = 0.0
        end

        net.layers[i] = {
            W = W, b = b,
            vW = vW, vb = vb,
            act = act, fan_in = fan_in, fan_out = fan_out
        }
    end

    return setmetatable(net, { __index = nn })
end

-- ─── Forward pass ─────────────────────────────────────────────────────────────

function nn:forward(input, dropout_rate)
    local cache = { [0] = { a = input } }
    local cur   = input

    for i = 1, self.n_layers do
        local layer  = self.layers[i]
        local act_fn = ACTS[layer.act].f
        local z, a   = {}, {}
        local mask   = {}

        for o = 1, layer.fan_out do
            local sum = layer.b[o]
            for inp = 1, layer.fan_in do
                sum = sum + layer.W[o][inp] * cur[inp]
            end
            z[o] = sum
            a[o] = act_fn(sum)
        end

        -- Dropout on hidden layers during training only
        if dropout_rate and dropout_rate > 0 and i < self.n_layers then
            local keep = 1.0 - dropout_rate
            for o = 1, layer.fan_out do
                if math.random() < keep then
                    mask[o] = 1.0 / keep  -- inverted dropout: scale up to maintain expected value
                else
                    mask[o] = 0.0
                    a[o]    = 0.0
                end
            end
        else
            for o = 1, layer.fan_out do mask[o] = 1.0 end
        end

        -- Last layer softmax override
        if i == self.n_layers and self.use_softmax then
            a = softmax(z)
            for o = 1, layer.fan_out do mask[o] = 1.0 end
        end

        cache[i] = { z = z, a = a, mask = mask }
        cur = a
    end

    return cache
end

function nn:predict(input)
    local cache = self:forward(input, 0)  -- no dropout at inference
    return cache[self.n_layers].a
end

-- ─── Backward pass (momentum SGD + gradient clipping) ─────────────────────────

local GRAD_CLIP = 5.0

function nn:backward(cache, target, lr, momentum)
    momentum = momentum or 0.0
    local L  = self.n_layers

    local out   = cache[L].a
    local act_d = ACTS[self.layers[L].act].d
    local delta = {}
    for o = 1, #out do
        delta[o] = (out[o] - target[o]) * act_d(out[o])
    end

    for i = L, 1, -1 do
        local layer  = self.layers[i]
        local a_prev = cache[i - 1].a
        local mask   = cache[i].mask

        -- Apply dropout mask to delta
        for o = 1, layer.fan_out do
            delta[o] = delta[o] * mask[o]
        end

        local delta_prev = {}
        if i > 1 then
            local act_d_prev = ACTS[self.layers[i - 1].act].d
            for inp = 1, layer.fan_in do
                local s = 0.0
                for o = 1, layer.fan_out do
                    s = s + layer.W[o][inp] * delta[o]
                end
                delta_prev[inp] = s * act_d_prev(a_prev[inp])
            end
        end

        -- Update weights with momentum + gradient clipping
        for o = 1, layer.fan_out do
            local d = delta[o]
            -- Clip gradient
            if d > GRAD_CLIP then d = GRAD_CLIP elseif d < -GRAD_CLIP then d = -GRAD_CLIP end

            for inp = 1, layer.fan_in do
                local g = d * a_prev[inp]
                layer.vW[o][inp] = momentum * layer.vW[o][inp] + lr * g
                layer.W[o][inp]  = layer.W[o][inp] - layer.vW[o][inp]
            end
            layer.vb[o] = momentum * layer.vb[o] + lr * d
            layer.b[o]  = layer.b[o] - layer.vb[o]
        end

        delta = delta_prev
    end
end

-- ─── Training loop with LR decay + dropout ────────────────────────────────────

function nn:train(data, epochs, lr, opts)
    opts = opts or {}
    local verbose      = opts.verbose      or false
    local momentum     = opts.momentum     or 0.9
    local dropout_rate = opts.dropout_rate or 0.0
    local decay        = opts.decay        or 1.0    -- multiply lr by this each epoch
    local min_lr       = opts.min_lr       or 1e-5

    local cur_lr = lr

    for epoch = 1, epochs do
        local total_loss = 0.0
        -- Shuffle
        for i = #data, 2, -1 do
            local j = math.random(i)
            data[i], data[j] = data[j], data[i]
        end

        for _, sample in ipairs(data) do
            local cache = self:forward(sample.input, dropout_rate)
            local out   = cache[self.n_layers].a
            local loss  = 0.0
            for k = 1, #out do
                loss = loss + (out[k] - sample.target[k]) ^ 2
            end
            total_loss = total_loss + loss / #out
            self:backward(cache, sample.target, cur_lr, momentum)
        end

        -- LR decay
        if decay < 1.0 then
            cur_lr = math.max(min_lr, cur_lr * decay)
        end

        if verbose and (epoch % verbose == 0) then
            io.write(string.format("  Epoch %4d | Loss: %.6f | LR: %.6f\n",
                epoch, total_loss / #data, cur_lr))
            io.flush()
        end
    end
end

-- ─── Weight persistence ───────────────────────────────────────────────────────

function nn:save(path)
    local lines = {}
    lines[#lines + 1] = string.format("layers=%d", self.n_layers)
    for i, layer in ipairs(self.layers) do
        lines[#lines + 1] = string.format("layer=%d fan_in=%d fan_out=%d act=%s",
            i, layer.fan_in, layer.fan_out, layer.act)
        for o = 1, layer.fan_out do
            local row = {}
            for inp = 1, layer.fan_in do
                row[#row + 1] = string.format("%.10f", layer.W[o][inp])
            end
            lines[#lines + 1] = "W " .. table.concat(row, " ")
        end
        for o = 1, layer.fan_out do
            lines[#lines + 1] = string.format("b %.10f", layer.b[o])
        end
    end
    local f = assert(io.open(path, "w"))
    f:write(table.concat(lines, "\n") .. "\n")
    f:close()
end

function nn.load(path)
    local f = assert(io.open(path, "r"), "Cannot open weights: " .. path)
    local lines = {}
    for line in f:lines() do lines[#lines + 1] = line end
    f:close()

    local net      = { layers = {}, n_layers = 0 }
    local idx      = 1
    local n_layers = tonumber(lines[idx]:match("layers=(%d+)"))
    net.n_layers   = n_layers
    idx            = idx + 1

    for i = 1, n_layers do
        local hdr     = lines[idx]; idx = idx + 1
        local fan_in  = tonumber(hdr:match("fan_in=(%d+)"))
        local fan_out = tonumber(hdr:match("fan_out=(%d+)"))
        local act     = hdr:match("act=(%a+)")
        local W, b    = {}, {}
        local vW, vb  = {}, {}

        for o = 1, fan_out do
            local row  = {}
            local vrow = {}
            local nums = lines[idx]:sub(3)
            for v in nums:gmatch("[%d%.%-%+eE]+") do
                row[#row + 1]  = tonumber(v)
                vrow[#vrow + 1] = 0.0
            end
            W[o]  = row
            vW[o] = vrow
            idx   = idx + 1
        end
        for o = 1, fan_out do
            b[o]  = tonumber(lines[idx]:match("b (.+)"))
            vb[o] = 0.0
            idx   = idx + 1
        end

        net.layers[i] = { W = W, b = b, vW = vW, vb = vb,
                          act = act, fan_in = fan_in, fan_out = fan_out }
    end

    return setmetatable(net, { __index = nn })
end

-- ─── Utilities ───────────────────────────────────────────────────────────────

function nn:argmax(vec)
    local best_i, best_v = 1, vec[1]
    for i = 2, #vec do
        if vec[i] > best_v then best_i, best_v = i, vec[i] end
    end
    return best_i, best_v
end

return nn