package main

import (
	"fmt"
	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/llgcode/draw2d/draw2dkit"
	"image"
	"image/color"
	"os"
)

const (
	FRAMESDIR = "draw2dimg_01_frames"
)

func init() {
	os.Mkdir(FRAMESDIR, 0777)
}

func main() {
	os.Chdir(FRAMESDIR)
	fmt.Println(os.Getwd())
	for i := 1; i <= 20; i++ {
		// Initialize the graphic context on an RGBA image
		frame := image.NewRGBA(image.Rect(0, 0, 300, 300))
		gc := draw2dimg.NewGraphicContext(frame)

		// Draw background
		gc.SetFillColor(color.RGBA{0x00, 0x00, 0x00, 0xff})
		draw2dkit.Rectangle(gc, 0, 0, 300, 300)

		// Set some properties
		gc.SetFillColor(color.RGBA{0xff, 0xff, 0xff, 0xff})
		gc.SetStrokeColor(color.RGBA{0x00, 0x00, 0x00, 0xff})
		gc.SetLineWidth(1)

		// Draw a closed shape
		var v float64 = float64(i * 10)
		draw2dkit.Rectangle(gc, v, v, 100+v, 100+v)
		gc.FillStroke()

		// Save to file
		var filename string = fmt.Sprintf("%06d.png", i)
		fmt.Println(filename)
		draw2dimg.SaveToPngFile(filename, frame)
	}
}
