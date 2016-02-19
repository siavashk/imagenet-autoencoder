require 'torch'
require 'cutorch'
require 'cudnn'
require 'nn'
require 'cunn'

autoencoder = {}
autoencoder.__index = autoencoder

setmetatable(autoencoder, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function autoencoder.new()
  local self = setmetatable({}, autoencoder)
  return self
end

function autoencoder:initialize()
  local pool_layer1 = nn.SpatialMaxPooling(2, 2, 2, 2)
  local pool_layer2 = nn.SpatialMaxPooling(2, 2, 2, 2)

  self.net = nn.Sequential()
  self.net:add(nn.SpatialConvolution(3, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(pool_layer1)
  self.net:add(nn.SpatialConvolution(12, 24, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(pool_layer2)
  self.net:add(nn.Reshape(24 * 14 * 14))
  self.net:add(nn.Linear(24 * 14 * 14, 1568))
  self.net:add(nn.Linear(1568, 24 * 14 * 14))
  self.net:add(nn.Reshape(24, 14, 14))
  self.net:add(nn.SpatialConvolution(24, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxUnpooling(pool_layer2))
  self.net:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxUnpooling(pool_layer1))
  self.net:add(nn.SpatialConvolution(12, 3, 3, 3, 1, 1, 0, 0))

  self.net = self.net:cuda()
end

function autoencoder:printself()
  print(self.net)
end
