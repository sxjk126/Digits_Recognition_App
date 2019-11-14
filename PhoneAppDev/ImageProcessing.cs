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
            /*
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
            */
            Example();
        }
        class Digit
        {
            [VectorType(785)] public float[] PixelValues;
        }

        public static class ExtractPixels
        {
            // Sample that loads the images from the file system, resizes them (
            // ExtractPixels requires a resizing operation), and extracts the values of
            // the pixels as a vector. 
            public static void Example()
            {
                // Create a new ML context, for ML.NET operations. It can be used for
                // exception tracking and logging, as well as the source of randomness.
                var mlContext = new MLContext();

                // Downloading a few images, and an images.tsv file, which contains a
                // list of the files from the dotnet/machinelearning/test/data/images/.
                // If you inspect the fileSystem, after running this line, an "images"
                // folder will be created, containing 4 images, and a .tsv file
                // enumerating the images. 
                var imagesDataFile = Path.Combine("Resources", "Images",
                                     @"D:\Repos\Number_Two.jpg");

                // Preview of the content of the images.tsv file
                //
                // imagePath    imageType
                // tomato.bmp   tomato
                // banana.jpg   banana
                // hotdog.jpg   hotdog
                // tomato.jpg   tomato

                var data = mlContext.Data.CreateTextLoader(new TextLoader.Options()
                {
                    Columns = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                }
                }).Load(imagesDataFile);

                var imagesFolder = Path.GetDirectoryName(imagesDataFile);
                // Image loading pipeline. 
                var pipeline = mlContext.Transforms.LoadImages("ImageObject",
                    imagesFolder, "ImagePath")
                    .Append(mlContext.Transforms.ResizeImages("ImageObjectResized",
                        inputColumnName: "ImageObject", imageWidth: 28, imageHeight:
                        28))
                    .Append(mlContext.Transforms.ExtractPixels("Pixels",
                        "ImageObjectResized"));

                var transformedData = pipeline.Fit(data).Transform(data);

                // Preview the transformedData. 
                PrintColumns(transformedData);

            }

            private static void PrintColumns(IDataView transformedData)
            {
                Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} {4, -25}",
                    "ImagePath", "Name", "ImageObject", "ImageObjectResized", "Pixels");

                using (var cursor = transformedData.GetRowCursor(transformedData
                    .Schema))
                {
                    // Note that it is best to get the getters and values *before*
                    // iteration, so as to faciliate buffer sharing (if applicable), and
                    // column -type validation once, rather than many times.

                    ReadOnlyMemory<char> imagePath = default;
                    ReadOnlyMemory<char> name = default;
                    Bitmap imageObject = null;
                    Bitmap resizedImageObject = null;
                    VBuffer<float> pixels = default;

                    var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                        .Schema["ImagePath"]);

                    var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                        .Schema["Name"]);

                    var imageObjectGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                        "ImageObject"]);

                    var resizedImageGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                        "ImageObjectResized"]);

                    var pixelsGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[
                        "Pixels"]);

                    while (cursor.MoveNext())
                    {

                        imagePathGetter(ref imagePath);
                        nameGetter(ref name);
                        imageObjectGetter(ref imageObject);
                        resizedImageGetter(ref resizedImageObject);
                        pixelsGetter(ref pixels);

                        Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} " +
                            "{4, -25}", imagePath, name, imageObject.PhysicalDimension,
                            resizedImageObject.PhysicalDimension, string.Join(",",
                            pixels.DenseValues().Take(5)) + "...");
                    }

                    // Dispose the image.
                    imageObject.Dispose();
                    resizedImageObject.Dispose();
                }
            }
        }

    }
}