package main

import (
	"fmt"
	"image"
	"image/color"
	"image/font"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
)

// set constants ('E' is a math const; remove '/rand' on import when using)
const (
	NUMINPUTNODES  = 2
	NUMHIDDENNODES = 2
	NUMOUTPUTNODES = 1
	NUMNODES       = NUMINPUTNODES + NUMHIDDENNODES + NUMOUTPUTNODES
	ARRAYSIZE      = NUMNODES + 1
	MAXITERATIONS  = 100000
	LEARNINGRATE   = 0.2

	FRAMESDIR  = "shaffer_frames"
	CANVASSIDE = 500
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
		fmt.Println(iteration)
		trainingExample(iteration, values, expectedValues)
		fmt.Println("training inputs: ")
		for i := range values {
			fmt.Printf("%d ", values[i])
		}
		fmt.Println("\nneuron outputs: ")
		activateNetwork(weights, values, thresholds)
		for i := range values {
			fmt.Printf("%d ", values[i])
		}
		fmt.Println("\nevaluate errors and adjust network: ")
		sumOfSquaredErrors := updateWeights(weights, values, expectedValues, thresholds)
		for i := range values {
			fmt.Printf("%d ", values[i])
		}
		fmt.Println("\nsum of squared errors: ", sumOfSquaredErrors)

		if iteration%101 == 0 {
			displayNetwork(values, iteration, sumOfSquaredErrors)
		}
	}
}

// set 0.0 values to all slice cells
func initialize(weights [][]float64, values []float64, expectedValues []float64, thresholds []float64) {
	for x := 0; x <= NUMNODES; x++ {
		values[x] = 0.0
		expectedValues[x] = 0.0
		thresholds[x] = 0.0
		weights[x] = make([]float64, ARRAYSIZE)

		for y := 0; y <= NUMNODES; y++ {
			weights[x][y] = 0.0
		}
	}
}

// set random weights of all possible node-to-node connections
// set random threshold levels on the actual member neurons of network
func connectNodes(r *rand.Rand, weights [][]float64, thresholds []float64) {
	for x := 1; x <= NUMNODES; x++ {
		for y := 1; y <= NUMNODES; y++ {
			weights[x][y] = r.Float64() * 2
		}
	}

	thresholds[3] = r.Float64()
	thresholds[4] = r.Float64()
	thresholds[5] = r.Float64()

	//fmt.Printf("%d %d\n%d %d\n%d\n%d\n%d %d %d\n", weights[1][3], weights[1][4], weights[2][3], weights[2][4], weights[3][5], weights[4][5], thresholds[3], thresholds[4], thresholds[5])
}

// set XOR values of inputs (values[1], values[2])
// and expected output (values[5]) to train network against
func trainingExample(iter int, values []float64, expectedValues []float64) {
	switch iter % 4 {
	case 0:
		values[1] = 1.0
		values[2] = 1.0
		expectedValues[5] = 0.0
	case 1:
		values[1] = 0.0
		values[2] = 1.0
		expectedValues[5] = 1.0
	case 2:
		values[1] = 1.0
		values[2] = 0.0
		expectedValues[5] = 1.0
	case 3:
		values[1] = 0.0
		values[2] = 0.0
		expectedValues[5] = 0.0
	}
}

// evaluate inputs through the network to its output
// nodes are summed in each node by weights
// then their sums evaluated against sigmoid
func activateNetwork(weights [][]float64, values []float64, thresholds []float64) {
	// evaluate inputs (including hidden nodes)
	for hidden := 1 + NUMINPUTNODES; hidden < 1+NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
		weightedInput := 0.0
		for input := 1; input < 1+NUMINPUTNODES; input++ {
			weightedInput += weights[input][hidden] * values[input]
		}

		weightedInput -= thresholds[hidden]
		values[hidden] = 1.0 / (1.0 + math.Pow(math.E, -weightedInput))
	}

	// evaluate outputs (including hidden nodes)
	for output := 1 + NUMINPUTNODES + NUMHIDDENNODES; output < 1+NUMNODES; output++ {
		weightedInput := 0.0
		for hidden := 1 + NUMINPUTNODES; hidden < 1+NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
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

	for output := 1 + NUMINPUTNODES + NUMHIDDENNODES; output < 1+NUMNODES; output++ {
		absoluteError := expectedValues[output] - values[output]
		sumOfSquaredErrors += math.Pow(absoluteError, 2)
		outputErrorGradient := values[output] * (1.0 - values[output]) * absoluteError

		for hidden := 1 + NUMINPUTNODES; hidden < 1+NUMINPUTNODES+NUMHIDDENNODES; hidden++ {
			delta := LEARNINGRATE * values[hidden] * outputErrorGradient
			weights[hidden][output] += delta
			hiddenErrorGradient := values[hidden] * (1.0 - values[hidden]) * outputErrorGradient * weights[hidden][output]

			for input := 1; input < 1+NUMINPUTNODES; input++ {
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

func displayNetwork(values []float64, iteration int, sumOfSquaredErrors float64) {
	frame := image.NewRGBA(image.Rect(0, 0, CANVASSIDE, CANVASSIDE))
	gc := draw2dimg.NewGraphicContext(frame)

	outputColorVal := mapFloatToInt(values[5], 0, 1, 0, 255)
	fmt.Println(outputColorVal)
	gc.SetFillColor(color.Gray{uint8(outputColorVal)})
	draw2dkit.Rectangle(gc, 0, 0, CANVASSIDE, CANVASSIDE)
	gc.Fill()

	input1ColorVal := mapFloatToInt(values[1], 0, 1, 0, 255)
	fmt.Println(input1ColorVal)
	gc.SetFillColor(color.Gray{uint8(input1ColorVal)})
	draw2dkit.Rectangle(gc, 100, 100, 400, 225)
	gc.Fill()

	input2ColorVal := mapFloatToInt(values[2], 0, 1, 0, 255)
	fmt.Println(input2ColorVal)
	gc.SetFillColor(color.Gray{uint8(input2ColorVal)})
	draw2dkit.Rectangle(gc, 100, 275, 400, 400)
	gc.Fill()

	var info string = fmt.Sprintf("Iteration %06d, SSE %d", iteration, sumOfSquaredErrors)
	infoColorVal := 255 - outputColorVal
	gc.SetStrokeColor(color.Gray{uint8(infoColorVal)})
	gc.StrokeString(info)
	gc.Stroke()

	var filename string = fmt.Sprintf("%06d.png", iteration)
	fmt.Println(filename)
	draw2dimg.SaveToPngFile(filename, frame)
}

func mapFloatToInt(value, oldMin, oldMax, newMin, newMax float64) int {
	proportion := (value - oldMin) / (oldMax - oldMin)
	newValue := int(proportion*(newMax-newMin) + newMin)
	return newValue
}
