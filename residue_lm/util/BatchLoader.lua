local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_dir, batch_size, seq_length, split_fractions)

    local self = {}
    setmetatable(self, BatchLoader)

	-- preprocess the data
    local input_file = path.join(data_dir, 'residue_lstm_input.txt')
    local tensor_file = path.join(data_dir, 'data.t7')
  	if not path.exists(tensor_file) then 
        BatchLoader.text_to_tensor(input_file,tensor_file)
    end

	-- load the data
    local data, vocab = unpack(torch.load(tensor_file))
    self.vocab_mapping = vocab


	-- slice the data so that we get evenly sized batches
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
    end

	-- find out how many letters we're dealing with
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    self.batch_size = batch_size
    self.seq_length = seq_length

	-- set outputs as the inputs, shifted by one
	-- letter to predict at t == letter at t+1
    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]

	-- split into batches of appropriate size	
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
    assert(#self.x_batches == #self.y_batches)

    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
	self.ntest= 0
    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

-- reset to the beginning of the batch set
function BatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

-- generate the next batch of data
function BatchLoader:next_batch(split_index)
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end

    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end

-- preprocess the text data into tensors for Torch
function BatchLoader.text_to_tensor(in_textfile, out_tensorfile)
   
	local data=readAll(in_textfile)
	local vocab, count={}, 1
	local data_tensor={}
	for char in data:gmatch'.' do
		if vocab[char]==nil then
			vocab[char]=count
			count= count+1
		end
		table.insert(data_tensor, vocab[char])
	end
	local data_tensor = torch.ByteTensor(data_tensor)

	torch.save(out_tensorfile, {data_tensor, vocab})
end

return BatchLoader

