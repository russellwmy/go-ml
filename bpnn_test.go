package goml
import (
	"testing"
)
func TestBPNN(m *testing.T) {
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