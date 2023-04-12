// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import nsfw from "nsfwjs";
import tf from "@tensorflow/tfjs-node";
import axios from "axios";

export default async function handler(req, res) {
  const pic = await axios.get(
    `https://preview.redd.it/hykodwxnvita1.jpg?width=960&crop=smart&auto=webp&v=enabled&s=1e9ebb5d4f5796a43cfac1ce73fb224f9e01b712`,
    {
      responseType: "arraybuffer",
    }
  );
  const model = await nsfw.load(); // To load a local model, nsfw.load('file://./path/to/model/')
  // Image must be in tf.tensor3d format
  // you can convert image to tf.tensor3d with tf.node.decodeImage(Uint8Array,channels)
  const image = await tf.node.decodeImage(pic.data, 3);
  const predictions = await model.classify(image);
  image.dispose(); // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).

  return res.status(200).json({ ...predictions });
}
