using System;
using System.IO;
using System.Drawing;
using GrapeCity.Documents.Imaging;
using GrapeCity.Documents.Text;
using Newtonsoft.Json;
using Microsoft.ML;

namespace CSharpImageProcessing
{
    class Program
    {
        static void Main(string[] args)
        {
            
            //Get the image path
            var origImagePath = Path.Combine("Resources", "Images",
                                     @"D:\Repos\Number_Two.jpg");

            //Initialize GcBitmap
            GcBitmap origBmp = new GcBitmap();

            
            //Load image from file
            origBmp.Load(origImagePath);

            
            //Convert to Grayscale
            GrayscaleBitmap gsBmp = new GrayscaleBitmap(origBmp.PixelWidth, origBmp.PixelHeight);
            gsBmp = origBmp.ToGrayscaleBitmap(whiteIsZero: false);
            gsBmp.ToGcBitmap();

            GcBitmap grayBmp = new GcBitmap();
            grayBmp = gsBmp.ToGcBitmap();
            //Reduce image 
            int rwidth = 28;
            int rheight = 28;
            GcBitmap smallBmp = grayBmp.Resize(rwidth, rheight,
                                InterpolationMode.Downscale);

            //Save scaled image to file
            smallBmp.SaveAsJpeg(@"D:\Repos\ResizedTwo.jpg");
        }


    }
}