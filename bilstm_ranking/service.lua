require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util/misc'
require 'util.OneHot'

cmd = torch.CmdLine()
cmd:argument('-model', 'model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid', -1, '>0 for GPU, -1 for CPU')
cmd:option('-port', 8080, 'port to run service on')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid>=0 then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(opt.gpuid)
end


-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end

local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state
local model = checkpoint.opt.model
local num_layers = checkpoint.opt.num_layers
local init_state = {}
local h_init = torch.zeros(1, checkpoint.opt.rnn_size):float()

if opt.gpuid>=0 then
	h_init=h_init:cuda()
	protos.fwd:cuda()
	protos.bwd:cuda()
	protos.mic_estimator:cuda()
	protos.mic_criterion:cuda()
end

for L=1,checkpoint.opt.num_layers do
	table.insert(init_state, h_init:clone())
	table.insert(init_state, h_init:clone())
end
local state_size = #init_state

-- make sure dropout works properly
protos.fwd:evaluate()
protos.bwd:evaluate()
protos.mic_estimator:evaluate()
	
function get_pred(seed_text)
	local seq_length=seed_text:len()
	local fwd_state, bwd_state = {[0] = init_state}, {[seq_length+1] = init_state}

	local x={}
	for t=1,seq_length do
		x[t]=torch.Tensor({vocab[string.char(seed_text:byte(t))]}):float()
	end
	if opt.gpuid>=0 then
        for t=1, seq_length do x[t] = x[t]:cuda() end
    end


	local fwdh_out, bwdh_out
    for t=1, seq_length do
		local _t=seq_length-t+1

	    local fwdlst = protos.fwd:forward{x[t], unpack(fwd_state[t-1])}
		local bwdlst = protos.bwd:forward{x[_t], unpack(bwd_state[_t+1])}

		fwdh_out, bwdh_out=fwdlst[#fwdlst]:clone(), bwdlst[#bwdlst]:clone()
	    fwd_state[t], bwd_state[_t] = {}, {}
	    for i=1,#init_state do
			table.insert(fwd_state[t], fwdlst[i]) 
			table.insert(bwd_state[_t], bwdlst[i])
		end
	end

	local prediction=protos.mic_estimator:forward({fwdh_out, bwdh_out})
	return prediction
end



local app = require('waffle')

app.get('/blstm/(%a+)', function(req, res)
	local seq = req.params[1]
	if seq:find('X')==nil and seq:find('B')==nil and seq:find('U')==nil then --ignore strange sequences
		pred = get_pred(seq)[1][1]
		res.send(string.format('%.5f -- %s', pred, seq))
	else
		res.send('Invalid sequence (may contain X,B,U?)')
	end   
end)

app.listen({port=opt.port})





