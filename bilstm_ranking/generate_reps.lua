require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-test_file', 'data/amp/input.seqs')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()


-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    --print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end

local checkpoint = torch.load(opt.model)
local init_state = {}
for L=1,checkpoint.opt.num_layers do
	local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
	if opt.gpuid >= 0 then h_init = h_init:cuda() end
	table.insert(init_state, h_init:clone())
	table.insert(init_state, h_init:clone())
end

local protos = checkpoint.protos

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state
local model = checkpoint.opt.model

local num_layers = checkpoint.opt.num_layers
local state_size = #init_state

protos.fwd:evaluate()
protos.bwd:evaluate()
protos.mic_estimator:evaluate()
	
function get_pred(seed_text)
	local seq_length=seed_text:len()
	local fwd_state, bwd_state = {[0] = init_state}, {[seq_length+1] = init_state}
	local x={}
	for t=1,seq_length do
		x[t]=torch.Tensor({vocab[string.char(seed_text:byte(t))]})
	end

	local fwdh_out, bwdh_out
    for t=1,seq_length do
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

	return torch.cat(fwdh_out, bwdh_out,2):double()
end

local reprs={}
reprs.AMP={}
local line_no=1
for seq in io.lines(opt.test_file) do

	local prot=seq:split('\t')[1]
	if prot:find('X')==nil and prot:find('B')==nil then
		reprs.AMP[prot..':'..seq:split('\t')[3]]=get_pred(prot):squeeze():clone()
	end
	line_no=line_no+1
end

torch.save('../../vis/amp2.t7',reprs)

