module FluxTest

using CUDA
using Flux
using MLDatasets: MNIST


function create_data_loaders()
	train_x, train_y = MNIST.traindata(Float32)
	test_x, test_y = MNIST.testdata(Float32)
	train_x, test_x = reshape(train_x, 28, 28, 1, :), reshape(test_x, 28, 28, 1, :)
	train_y, test_y = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9)
	train = Flux.DataLoader((train_x, train_y), batchsize = 128, shuffle = true)
	test = Flux.DataLoader((test_x, test_y), batchsize = 128)
	return train, test	
end

#equivalent to cat(x, y, z...; dims = 3)
cat_channels_2d(xy...) = cat(xy...; dims = Val(3))
#equivalent to cat(x, z...; dims = 2)
cat_channels_1d(x...) = cat(x...; dims = Val(2))

function conv_module_a(imgs_in, imgs_out)
	return Chain(
		Conv((3, 3), imgs_in=>imgs_in, relu, pad=SamePad()),
		Conv((3, 3), imgs_in=>imgs_out, relu, pad=SamePad()),
		Conv((3, 3), imgs_out=>imgs_out, relu, pad=SamePad()),
	) 	
end

function shell_downsample(conv_mod, imgs_pair_skip, imgs_pair_long) 
	short_conv = Conv((2, 2), imgs_pair_skip, relu, pad = SamePad(), stride = (2, 2))
	long_chain = Chain(
		conv_mod,
		Conv((2, 2), imgs_pair_long, relu, pad = SamePad(), stride = (2, 2)),
	)
	return Parallel(cat_channels_2d, short_conv, long_chain)
end

function create_inception()
	return Chain(
		Conv((5,5), 1=>8, relu), #24, 8
		shell_downsample(conv_module_a(8, 8), 8=>4, 8=>12),#12, 16
		Dropout(0.25),
		shell_downsample(conv_module_a(16, 16), 16=>6, 16=>26),#6, 32
		shell_downsample(conv_module_a(32, 32), 32=>8, 32=>56),#3, 64
		Dropout(0.25),
		Flux.flatten,
		Dense(3*3*64=>512, relu),
		Dense(512=>128, relu),
		Dropout(0.25),
		Dense(128=>10),
	)
end




function conv_mod_1d_a(imgs_in, imgs_out)
	return Chain(
		Conv((3), imgs_in=>imgs_out, pad=SamePad()),
		Conv((3), imgs_in=>imgs_out, pad=SamePad()),
		Conv((3), imgs_in=>imgs_out, pad=SamePad())
	)	
end

function shell_conv_1d(conv_module, imgs_pair_skip, imgs_pair_long)
	skip_conv = Conv((3, 1), imgs_pair_skip, relu, pad = SamePad(), stride = (2, 1))
	long = Chain(
		conv_mod_1d_a,
		Conv((3, 1), imgs_pair_long, relu, pad = SamePad(), stride = (2, 1))
	)
	return Parallel(cat_channels_1d, skip_conv, long)
end

function create_finance_algo()
	return Chain(

	)
end


function create_lenet()
	return Chain(
		Conv((5, 5), 1=>6, relu), #24 
		MaxPool((2, 2)), #12
		Conv((5, 5), 6=>16, relu), #8
		MaxPool((2, 2)), #4
		Flux.flatten,
		Dense(256=>128, relu),
		Dense(128=>64, relu),
		Dense(64=>10),
	)
end






function eval_accuracy(loader, model, device)
	loss = 0f0
	accuracy = 0
	num_total = 0 
	for (x, y) in loader
		x, y = x |> device, y |> device
		y_hat = model(x)
		loss += Flux.logitcrossentropy(y_hat, y) * size(x)[end]
		accuracy += sum(Flux.onecold(y_hat |> cpu) .== Flux.onecold(y |> cpu))
		num_total += size(x)[end]
	end

	return (loss = loss / num_total, accuracy = accuracy / num_total)
end



function train()
	@info "creating data loaders"
	train_loader, test_loader = create_data_loaders()
	@info "train count: $(train_loader.nobs), test count: $(test_loader.nobs)"

	@info "creating model"
	model = create_inception() |> gpu
	@info "model params $(sum(length, Flux.params(model)))"

	parameters = Flux.params(model)
	#optimizer = ADAMW(0.003, (.85, .99), 0.01)
	optimizer = Adam(0.0005)
	loss(x, y) = Flux.logitcrossentropy(model(x), y)
	
	@info "beginning training"
	for i in 1:100
		for (x, y) in train_loader
			x, y = x |> gpu, y |> gpu
			calculated_grad = Flux.gradient(() -> loss(x, y), parameters) 
			Flux.Optimise.update!(optimizer, parameters, calculated_grad)
		end

		if i % 1 == 0
			train_accuracy = eval_accuracy(train_loader, model, gpu)
			test_accuracy = eval_accuracy(test_loader, model, gpu)
			@info "epoch: $(i), train accuracy: $(train_accuracy), test accuracy: $(test_accuracy)"
		end	
	end	
	@info "end"
end

CUDA.versioninfo()
train()

end 
