require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(vocab_size, char_size, rnn_size, n, dropout)

	dropout = dropout or 0 

	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	for L = 1,n do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local x, input_size_L
	local outputs = {}

	for L = 1, n do
		local prev_h, prev_c = inputs[L*2+1], inputs[L*2]

		-- For first layer, embed or use onehot
		if L == 1 then  
			x = OneHot(vocab_size)(inputs[1])
			input_size_L = vocab_size

			-- x = nn.LookupTable(vocab_size, char_size)(inputs[1])
			-- input_size_L = char_size

		-- otherwise, use the previous layer's output
		else 			
			x = outputs[(L-1)*2] 
			if dropout > 0 then x = nn.Dropout(dropout)(x) end
			input_size_L = rnn_size
    	end

		local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)	-- input transformation
		local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)	-- hidden transformation
		local all_input_sums = nn.CAddTable()({i2h, h2h})		-- W^t*X + U^t*H + b 

		local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
		local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)  -- divide for gating purposes

		local in_gate = nn.Sigmoid()(n1)							-- sigma(W^t*X + U^t*H + b)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)
		local in_transform = nn.Tanh()(n4)							-- tanh(W^t*X + U^t*H + b)

		local next_c = nn.CAddTable()({								-- f_t*C_{t-1} + i_t*C_t  
		    nn.CMulTable()({forget_gate, prev_c}),
		    nn.CMulTable()({in_gate, in_transform})
		  })
		local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- o_t*tanh(C_t)
    
		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

  -----------------------------------------------------------------------------------
  return nn.gModule(inputs, outputs)
end

return LSTM

