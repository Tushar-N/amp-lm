require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
require 'util.OneHot'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-length',20000,'number of characters to sample')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid>=0 then
	require 'cunn'
	require 'cutorch'
	cutorch.setDevice(opt.gpuid)
end

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
local current_state = {}
local h_init = torch.zeros(1, checkpoint.opt.rnn_size)

if opt.gpuid>=0 then
	h_init=h_init:cuda()
	protos.rnn:cuda()
end

for L = 1,checkpoint.opt.num_layers do
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end

state_size = #current_state

local res_seq=table.concat(ivocab,''):gsub('\n','')
local seq_out=''
for _, temperature in pairs({0.1, 0.5, 1.0, 1.2, 1.5, 2.0}) do

	for n=1, 10 do

		-- uniform random seed residue
		local rand_res = torch.randperm(res_seq:len())[1]
		seed_text = res_seq:sub(rand_res, rand_res)

		-- do a few seeded timesteps
		print('seeding with ' .. seed_text)
		for c in seed_text:gmatch'.' do
			prev_char = torch.Tensor{vocab[c]}
			seq_out= seq_out.. ivocab[prev_char[1]]
			if opt.gpuid>=0 then prev_char = prev_char:cuda() end
			local lst = protos.rnn:forward{prev_char, unpack(current_state)}
			current_state = {}
			for i=1,state_size do table.insert(current_state, lst[i]) end
			prediction = lst[#lst]
		end

		-- start sampling
		for i=1, opt.length do

			-- sample log probabilities from the previous timestep
			prediction:div(temperature) -- scale by temperature
			local probs = torch.exp(prediction):squeeze()
			probs:div(torch.sum(probs)) -- renormalize so probs sum to one
			prev_char = torch.multinomial(probs:float(), 1):resize(1):float()

			-- forward the rnn for next character
			local lst = protos.rnn:forward{prev_char, unpack(current_state)}
			current_state = {}
			for i=1,state_size do table.insert(current_state, lst[i]) end
			prediction = lst[#lst] -- last element holds the log probabilities

			seq_out= seq_out.. ivocab[prev_char[1]]
		end
		seq_out= seq_out.. '\n'
		print (string.format('T: %f -- %d/10..', temperature, n)) 
	
	end

end

--torch.save('seq_out.t7', seq_out)

-- keep only unique sequences. Half of them are oversampled
local uniq_seq={}
for _, seq in pairs(seq_out:split('\n')) do
	if uniq_seq[seq]==nil and seq:len()>8 and seq:len()<21 then
		uniq_seq[seq]=1
	end
end

f_out=io.open('../clustal_pruning/sampled_seqs.txt', 'w')
for seq, _ in pairs(uniq_seq) do f_out:write(seq..'\n') end
f_out:close()







