.PHONY: convert run

convert:
	@echo "Compiling convert.cpp..."
	g++ ./convert.cpp -o convert
	@echo "Running convert..."
	./convert

run:
	chmod +x run2.sh && ./run2.sh
