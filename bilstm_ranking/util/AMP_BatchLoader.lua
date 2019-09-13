local AMP_BatchLoader = {}
AMP_BatchLoader.__index = AMP_BatchLoader

function AMP_BatchLoader.create(data_dir, batch_size, split_fractions)

    local self = {}
    setmetatable(self, AMP_BatchLoader)

	local data_file = path.join(data_dir, 'data.t7')
	-- construct a tensor with all the data
    -- if not (path.exists(data_file)) then
    --     print('one-time setup: preprocessing input text file ')
    --     self:text_to_tensor(data_dir)
    -- end
    self:text_to_tensor(data_dir)

	local X, Y, vocab= unpack(torch.load(data_file))
	self.vocab, self.ivocab=vocab, {}
	for k,v in pairs(self.vocab) do self.ivocab[v]=k end
	self.vocab_size=#self.ivocab
	self.max_seq_length=0
	self.batch_size = batch_size
	self.output_size= Y[1]:size(1)

	-- divide sequences nicely, ditch remaining (sorry)
	local x_batches, y_batches={}, {}
	local remaining_x, remaining_y, len_idx={}, {}, {}

	for i=1, #X do
		local len=X[i]:size(1)
		self.max_seq_length=math.max(self.max_seq_length, len)

		-- new batch
		if remaining_x[len]==nil then
			remaining_x[len]=torch.zeros(self.batch_size, len)
			remaining_y[len]=torch.zeros(self.batch_size, Y[i]:size(1))
			len_idx[len]=1
	    end

		remaining_x[len][len_idx[len]]:copy(X[i])
		remaining_y[len][len_idx[len]]:copy(Y[i])
		len_idx[len]=len_idx[len]+1

		-- save the batch, start fresh
		if len_idx[len]>self.batch_size then
			table.insert(x_batches, remaining_x[len])
			table.insert(y_batches, remaining_y[len])
			remaining_x[len], remaining_y[len]=nil, nil
			len_idx[len]=1
		end
	end

	self.x_batches= x_batches
	self.y_batches= y_batches
	self.nbatches= #self.x_batches

    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.ntest = 0

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function AMP_BatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function AMP_BatchLoader:next_batch(split_index)
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + test
    return self.x_batches[ix], self.y_batches[ix]
end



function AMP_BatchLoader:text_to_tensor(data_dir)
	
	--files for reading/writing
	local input_file = path.join(data_dir, 'bilstm_prune_input.txt')	
	local data_file = path.join(data_dir, 'data.t7')

	local vocab, vocab_idx={}, 1
	local sequences, sequence_data={}, {}

	for line in io.lines(input_file) do
		local seq, helicity, mic=unpack(line:split('\t'))
		mic = 1.0*mic/100.0 -- normalize

		local seq_tensor= {}
		for c in seq:gmatch'.' do
			if vocab[c]==nil then
				vocab[c]=vocab_idx
				vocab_idx=vocab_idx+1
			end
			table.insert(seq_tensor, vocab[c])
		end
		table.insert(sequences, torch.Tensor(seq_tensor))
		table.insert(sequence_data, torch.Tensor({mic}))
	end

	--print (min_hel, max_hel) --0.14	9	
	--print (min_mic, max_mic) --0.02	256	
    torch.save(data_file, {sequences, sequence_data, vocab})
	collectgarbage()
end

return AMP_BatchLoader

