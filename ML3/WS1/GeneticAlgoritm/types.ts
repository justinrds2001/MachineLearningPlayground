export interface Individual {
	a: number;
	b: number;
	c: number;
	d: number;
	e: number;
	f: number;
	[key: string]: number; // Add an index signature
}

export interface DataPoint {
	x: number;
	y: number;
}
