package goml

import (
	"fmt"
	"math"
	"math/rand"
)
func Rand (a float64, b float64) float64{
	return (b-a) * rand.Float64() + a
}

func Sigmoid (x float64) float64 {
	return math.Tanh(x)
}

func Dsigmoid(y float64) float64{
    return 1.0 - math.Pow(y, 2)
}

func Fill (a []float64, val float64){
	 for i, _ := range a {
	 	a[i] = val
	 }
}
func MakeMatrix(x int, y int) [][]float64{
	a := make ([][]float64, x)
	 for i := 0; i < x; i++ {
	 	a[i] = make ([]float64, y)
	 	for j := 0; j < y; j++ {
	 		a[i][j] = 0.0
	 	}
	}
	return a
}

type BPNN struct{
	ni int
	nh int
	no int
	ai []float64
	ah []float64
	ao []float64
	wi [][]float64
	wo [][]float64
	ci [][]float64
	co [][]float64
}

func (nn *BPNN) Init (ni int, nh int, no int) {
	
	nn.ni = ni + 1 
	nn.nh = nh
	nn.no = no

	nn.ai = make ([]float64, nn.ni)
	Fill(nn.ai, 1.0)
	nn.ah = make ([]float64, nn.nh)
	Fill(nn.ai, 1.0)
	nn.ao = make ([]float64, nn.no)
	Fill(nn.ai, 1.0)
	
	nn.wi = MakeMatrix (nn.ni, nn.nh)
	nn.wo = MakeMatrix (nn.nh, nn.no)

	for i := 0; i < len(nn.wi); i++ {
		for j := 0; j < len(nn.wi[i]); j++ {
			nn.wi[i][j] = Rand(-2.0, 2.0)
		}
	}

	for i := 0; i < len(nn.wo); i++ {
		for j := 0; j < len(nn.wo[i]); j++ {
			nn.wo[i][j] = Rand(-2.0, 2.0)
		}
	}
	nn.ci = MakeMatrix (nn.ni, nn.nh)
	nn.co = MakeMatrix (nn.nh, nn.no)
}

func (nn *BPNN) Update (inputs []float64) []float64{
	if len(inputs) != nn.ni-1 {
		fmt.Println("incorrect number of inputs")
	}

	// input activation
	for i := 0; i < (nn.ni-1); i++ {
		nn.ai[i] = inputs[i]
	}

	// hidden activation
	for j := 0; j < nn.nh; j++ {
		var sum float64
		for i := 0; i < nn.ni; i++ {
			sum = sum + nn.ai[i] * nn.wi[i][j]
		}
		nn.ah[j] = Sigmoid(sum)
	}

	// output activation
	for k := 0; k < nn.no; k++ {
		var sum float64
		for j := 0; j < nn.nh; j++ {
			sum = sum + nn.ah[j] * nn.wo[j][k]
		}
		nn.ao[k] = Sigmoid(sum)
	}
	return nn.ao

}

func (nn *BPNN) BackPropagate (targets []float64, N float64, M float64) float64{
	// calc output deltas
	// dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
	outputDeltas := make([]float64, nn.no)
	Fill(outputDeltas, 0.0)
	for k := 0; k < nn.no; k++ {
		error := targets[k] - nn.ao[k]
		outputDeltas[k] = error * Dsigmoid(nn.ao[k])
	}

	// update output weights
	for j := 0; j < nn.nh; j++ {
		for k := 0; k < nn.no; k++ {
			change :=  outputDeltas[k] * nn.ah[j]
			nn.wo[j][k] = nn.wo[j][k] + N * change + M * nn.co[j][k]
			nn.co[j][k] = change
		}
	}

	// calc hidden deltas
	hiddenDeltas := make([]float64, nn.nh)
	for j := 0; j < nn.nh; j++ {
		error := 0.0
		for k := 0; k < nn.no; k++ {
			error = error + outputDeltas[k] * nn.wo[j][k]
		}
		hiddenDeltas[j] = error * Dsigmoid(nn.ah[j])
	}
	
	// update input weights
	for i := 0; i < nn.ni; i++ {
		for j := 0; j < nn.nh; j++ {
			change :=  hiddenDeltas[j] * nn.ai[i]
			nn.wi[i][j] = nn.wi[i][j] + N * change + M * nn.ci[i][j]
			nn.ci[i][j] = change
		}
	}

	// calc combine error
	error := 0.0
	for k := 0; k < len(targets); k++ {
		x := 0.5 * (targets[k]-nn.ao[k])
		error = error + math.Pow(x, 2)
	}
	return error
}

func (nn *BPNN) Test (patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "pattern->", nn.Update(p[0]))
	}

}

func (nn *BPNN) Weights () {
	for _, w := range nn.wi {
		fmt.Println(w)
	}

	for _, w := range nn.wo {
		fmt.Println(w)
	}
	
}

func (nn *BPNN) Train (patterns [][][]float64, iterations int, N float64, M float64) {
	for i := 0; i < iterations; i++ {
		var error float64
		for _, p := range patterns {
			inputs := p[0]
			targets := p[1]
			nn.Update(inputs)
			error = error + nn.BackPropagate(targets, N, M)
		}
		if i % 100 == 0 {
			fmt.Println(error)
		}
	}
}
	
func main() {
	pat := [][][]float64 {
		{{0,0},{0}},
		{{0,1},{1}},
		{{1,0},{1}},
		{{1,1},{0}},
	}
	nn := BPNN{}
	nn.Init(2,2,1)
	nn.Train(pat, 1000, 0.3, 0.1)
	nn.Test(pat)
}
