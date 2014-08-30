void main()
{
	vec2 st = gl_TexCoord[0].st;
	vec4 color = texture2D(texture, st);
	vec4 colorBloom = texture2D(bloom, st);
	
	//bloomFactor = 1.6;

	// Add bloom to the image
	color += colorBloom * bloomFactor;
	
	gl_FragColor = color;
}