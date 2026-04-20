declare namespace JSX {
	type Element = any;

	interface IntrinsicAttributes {
		key?: string | number;
	}

	interface IntrinsicElements {
		box: any;
		col: any;
		divider: any;
		row: any;
		spacer: any;
		text: any;
	}
}
