package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
)

// set constants ('E' is a math const; remove '/rand' on import when using)
const (
	SQUAREROOTOFNODES = 10
	NUMINPUTNODES     = SQUAREROOTOFNODES * SQUAREROOTOFNODES
	NUMHIDDENNODES    = 20
	NUMOUTPUTNODES    = SQUAREROOTOFNODES * SQUAREROOTOFNODES
	NUMNODES          = NUMINPUTNODES + NUMHIDDENNODES + NUMOUTPUTNODES
	ARRAYSIZE         = NUMNODES
	MAXITERATIONS     = 2000000
	LEARNINGRATE      = 0.6

	FRAMESDIR  = "drawnn_frames"
	FRAMERATE  = 500
	CANVASSIDE = 500
	NODESQUARESIDE = CANVASSIDE / SQUAREROOTOFNODES
)

var (
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func init() {
	os.Mkdir(FRAMESDIR, 0777)
}

// initialize slices and call the init and connect functions on slices
// loop through network over MAXITERATIONS iterations,
// feedbacking inputs and expected outputs against actual outputs
func main() {
	fmt.Println("Neural Network Program")
	os.Chdir(FRAMESDIR)

	weights := make([][]float64, ARRAYSIZE)
	values := make([]float64, ARRAYSIZE)
	expectedValues := make([]float64, ARRAYSIZE)
	thresholds := make([]float64, ARRAYSIZE)

	initialize(weights, values, expectedValues, thresholds)
	connectNodes(r, weights, thresholds)

	for iteration := 0; iteration < MAXITERATIONS; iteration++ {
		trainingExample(r, values, expectedValues)
		activateNetwork(weights, values, thresholds)
		sumOfSquaredErrors := updateWeights(weights, values, expectedValues, thresholds)
		displayNetwork(values, iteration, sumOfSquaredErrors)
	}
}

// set 0.0 values to all slice cells
func initialize(weights [][]float64, values []float64, expectedValues []float64, thresholds []float64) {
	for x := 0; x < NUMNODES; x++ {
		values[x] = 0.0
		expectedValues[x] = 0.0
		thresholds[x] = 0.0
		weights[x] = make([]float64, ARRAYSIZE)

		for y := 0; y < NUMNODES; y++ {
			weights[x][y] = 0.0
		}
	}
}

// set random weights of all possible inner node-to-node connections
// set random threshold levels on the actual member neurons of network
func connectNodes(r *rand.Rand, weights [][]float64, thresholds []float64) {
	for x := 0; x < NUMNODES-NUMOUTPUTNODES; x++ {
		for y := NUMINPUTNODES; y < NUMNODES; y++ {
			weights[x][y] = r.Float64()
		}
	}

	for i := NUMINPUTNODES; i < NUMNODES; i++ {
		thresholds[i] = r.Float64()
	}
}

// set random binary values of inputs
// and equal corresponding expected output
// to train network against
func trainingExample(r *rand.Rand, values []float64, expectedValues []float64) {
	for i := 0; i < NUMINPUTNODES; i++ {
		if birand := r.Float64(); birand <= 0.5 {
			values[i], expectedValues[i+NUMINPUTNODES+NUMHIDDENNODES] = 0.0, 0.0
		} else {
			values[i], expectedValues[i+NUMINPUTNODES+NUMHIDDENNODES] = 1.0, 1.0
		}
	}
}

// evaluate inputs through the network to its output
// nodes are summed in each node by weights
// then their sums evaluated against sigmoid
func activateNetwork(weights [][]float64, values []float64, thresholds []float64) {
	// evaluate inputs (including hidden nodes)
	for hidden := NUMINPUTNODES; hidden < NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
		weightedInput := 0.0
		for input := 0; input < NUMINPUTNODES; input++ {
			weightedInput += weights[input][hidden] * values[input]
		}

		weightedInput -= thresholds[hidden]
		values[hidden] = 1.0 / (1.0 + math.Pow(math.E, -weightedInput))
	}

	// evaluate outputs (including hidden nodes)
	for output := NUMINPUTNODES + NUMHIDDENNODES; output < NUMNODES; output++ {
		weightedInput := 0.0
		for hidden := NUMINPUTNODES; hidden < NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
			weightedInput += weights[hidden][output] * values[hidden]
		}

		weightedInput -= thresholds[output]
		values[output] = 1.0 / (1.0 + math.Pow(math.E, -weightedInput))
	}
}

// evaluate all neuron values after running
// calculate errors against expectations
// adjust weights, thresholds accordingly
func updateWeights(weights [][]float64, values []float64, expectedValues []float64, thresholds []float64) float64 {
	sumOfSquaredErrors := 0.0

	for output := NUMINPUTNODES + NUMHIDDENNODES; output < NUMNODES; output++ {
		absoluteError := expectedValues[output] - values[output]
		sumOfSquaredErrors += math.Pow(absoluteError, 2)
		outputErrorGradient := values[output] * (1.0 - values[output]) * absoluteError

		for hidden := NUMINPUTNODES; hidden < NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
			delta := LEARNINGRATE * values[hidden] * outputErrorGradient
			weights[hidden][output] += delta
			hiddenErrorGradient := values[hidden] * (1.0 - values[hidden]) * outputErrorGradient * weights[hidden][output]

			for input := 0; input < NUMINPUTNODES; input++ {
				delta := LEARNINGRATE * values[input] * hiddenErrorGradient
				weights[input][hidden] += delta
			}
			thresholdDelta := LEARNINGRATE * -1 * hiddenErrorGradient
			thresholds[hidden] += thresholdDelta
		}
		delta := LEARNINGRATE * -1 * outputErrorGradient
		thresholds[output] += delta
	}
	return sumOfSquaredErrors
}

// generate the set of inputs and corresponding outputs
func displayNetwork(values []float64, iteration int, sumOfSquaredErrors float64) {
	if iteration%FRAMERATE == 0 {
		frame := image.NewRGBA(image.Rect(0, 0, CANVASSIDE*2, CANVASSIDE))
		gc := draw2dimg.NewGraphicContext(frame)
		gc.SetLineWidth(2)
		gc.SetStrokeColor(color.Black)

		gc.SetFillColor(color.White)
		draw2dkit.Rectangle(gc, 0, 0, CANVASSIDE*2, CANVASSIDE)
		gc.Fill()

			for x := 0; x < SQUAREROOTOFNODES; x++ {
				for y := 0; y < SQUAREROOTOFNODES; y++ {
					node := x * SQUAREROOTOFNODES + y
					inputColorVal := mapFloatToInt(values[node], 0, 1, 0, 255)
					outputColorVal := mapFloatToInt(values[node+NUMINPUTNODES+NUMHIDDENNODES], 0, 1, 0, 255)
					gc.SetFillColor(color.Gray{uint8(inputColorVal)})
					draw2dkit.Rectangle(gc, float64(x*NODESQUARESIDE), float64(y*NODESQUARESIDE), float64((x+1)*NODESQUARESIDE), float64((y+1)*NODESQUARESIDE))
					gc.FillStroke()

					gc.SetFillColor(color.Gray{uint8(outputColorVal)})
					draw2dkit.Rectangle(gc, float64(x*NODESQUARESIDE+CANVASSIDE), float64(y*NODESQUARESIDE), float64((x+1)*NODESQUARESIDE+CANVASSIDE), float64((y+1)*NODESQUARESIDE))
					gc.FillStroke()
				}
			}

		var filename string = fmt.Sprintf("%06d.png", iteration)
		fmt.Println(filename)
		draw2dimg.SaveToPngFile(filename, frame)
	}
}

func mapFloatToInt(value, oldMin, oldMax, newMin, newMax float64) int {
	proportion := (value - oldMin) / (oldMax - oldMin)
	newValue := int(proportion*(newMax-newMin) + newMin)
	return newValue
}
