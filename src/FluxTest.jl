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

function inception_mod_a(imgs_in, imgs_out)
	long_chain = Chain(
		Conv((3, 3), imgs_in=>imgs_out, relu),
		Conv((3, 3), imgs_out=>imgs_out, relu)
	)
	short_chain = Chain(
		Conv((1, 1), imgs_in=>imgs_out, relu)
	)
	return Parallel(long_chain, short_chain)
end

function create_model()
	#TODO: currently numbers will mis match, consider creating automatic generationg for numbers and layers
	return Chain(
		Conv((5, 5), 1=>6, relu), #SamePad()
		#inception_mod_a(6, 18),
		MaxPool((2, 2)), #14
		#inception_mod_a(18, 24),
		Conv((5, 5), 6=>16, relu),
		MaxPool((2, 2)), #7
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
	model = create_model() |> gpu
	@info "model params $(sum(length, Flux.params(model)))"

	parameters = Flux.params(model)
	optimizer = ADAM(0.0003)

	@info "beginning training"
	for i in 1:1000
		for (x, y) in train_loader
			x, y = x |> gpu, y |> gpu
			calculated_grad = Flux.gradient(parameters) do 
				y_hat = model(x)
				Flux.logitcrossentropy(y_hat, y)
			end	

			Flux.Optimise.update!(optimizer, parameters, calculated_grad)
		end

		if i % 10 == 0
			train_accuracy = eval_accuracy(train_loader, model, gpu)
			test_accuracy = eval_accuracy(test_loader, model, gpu)
			@info "epoch: $(i), train accuracy: $(train_accuracy), test accuracy: $(test_accuracy)"
		end	
	end	
	@info "end"
end

CUDA.versioninfo()
train()

end # module
