package main

import (
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"time"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
)

const (
	CANVASSIDE = 500
	FRAMESDIR  = "draw2d_02_frames"
	NUMFRAMES  = 200
	NUMSQUARES = 50
	SQUARESIDE = 50
)

var (
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func init() {
	os.Mkdir(FRAMESDIR, 0777)
}

func main() {
	os.Chdir(FRAMESDIR)

	squares := make([]int, NUMSQUARES)

	for i := 1; i <= NUMFRAMES; i++ {
		// Initialize the graphic context on an RGBA image
		frame := image.NewRGBA(image.Rect(0, 0, CANVASSIDE, CANVASSIDE))
		gc := draw2dimg.NewGraphicContext(frame)

		// Set each square color value
		setSquareColors(squares, r)

		// Draw background
		gc.SetFillColor(color.White)
		draw2dkit.Rectangle(gc, 0, 0, CANVASSIDE, CANVASSIDE)

		// Set some properties
		gc.SetLineWidth(1)

		// Draw squares
		for square := range squares {
			gc.SetFillColor(color.Gray{uint8(squares[square])})
			gc.SetStrokeColor(color.Black)
			var x float64 = mapFloat(r.Float64(), 0, 1, 0, CANVASSIDE)
			var y float64 = mapFloat(r.Float64(), 0, 1, 0, CANVASSIDE)
			draw2dkit.Rectangle(gc, x, y, SQUARESIDE+x, SQUARESIDE+y)
			gc.FillStroke()
		}

		// Save to file
		var filename string = fmt.Sprintf("%06d.png", i)
		fmt.Println(filename)
		draw2dimg.SaveToPngFile(filename, frame)
	}
}

func setSquareColors(squares []int, r *rand.Rand) {
	for i := range squares {
		squares[i] = mapFloatToInt(r.Float64(), 0, 1, 0, 255)
	}
}

func mapFloatToInt(value, oldMin, oldMax, newMin, newMax float64) int {
	proportion := (value - oldMin) / (oldMax - oldMin)
	newValue := int(proportion*(newMax-newMin) + newMin)
	return newValue
}

func mapFloat(value, oldMin, oldMax, newMin, newMax float64) float64 {
	proportion := (value - oldMin) / (oldMax - oldMin)
	newValue := proportion*(newMax-newMin) + newMin
	return newValue
}
