import turicreate as tc

sframe = tc.SFrame('images.sframe')

model = tc.object_detector.create(sframe, max_iterations=1200)

model.save('model.model')

# model.export_coreml('model.mlmodel')